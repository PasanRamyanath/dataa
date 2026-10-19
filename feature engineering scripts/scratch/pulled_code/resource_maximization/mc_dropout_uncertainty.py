"""
Multi-level MC-dropout + deep-ensemble + light test-time-augmentation
uncertainty estimation (§13.3, plan-marked OPTIONAL). Produces a per-tick
spread/uncertainty signal intended to feed fusion/calibration.py's
conformal calibration as an ADDITIONAL input downstream — this module
never imports or modifies fusion/calibration.py, matching the non-invasive
integration pattern already used for streaming/live_stream_glue.py's
on_tick hook (calibration.py's own caller decides whether/how to use the
extra signal; this module only produces it).

Two intensity levels, matching the plan's own "cheap vs expensive" framing
so a live loop can run the cheap version every tick and the expensive
version only periodically:
  - cheap:     config.MC_DROPOUT_CHEAP_PASSES passes at config.MC_DROPOUT_CHEAP_RATE
  - expensive: config.MC_DROPOUT_EXPENSIVE_PASSES passes at config.MC_DROPOUT_EXPENSIVE_RATE

`enable_mc_dropout()` forces only nn.Dropout submodules into train() mode
while leaving everything else (LayerNorm, GRU, Mamba block internals) in
eval() mode — running the WHOLE model in train() mode would also perturb
LayerNorm/BatchNorm statistics, corrupting the very predictions being
sampled, which is why this needs its own helper rather than a plain
`model.train()`.

Deep-ensemble spread (`ensemble_spread`) and light test-time augmentation
(`tta_predict`, jitter std config.TEST_TIME_AUGMENTATION_JITTER_STD) are
additional, independent uncertainty signals the plan lists alongside
MC-dropout — combined here by `combined_uncertainty` into one scalar per
tick, but each is also returned individually so a caller can inspect them
separately.

Never run against a real trained checkpoint here — every function is
exercised in this repo's own tests only against a tiny synthetic dropout
model, per this project's Kaggle-training-excluded-for-now scope.

Usage:
    from resource_maximization.mc_dropout_uncertainty import enable_mc_dropout, mc_dropout_predict
    mean, spread = mc_dropout_predict(model, x, passes=config.MC_DROPOUT_CHEAP_PASSES, rate=config.MC_DROPOUT_CHEAP_RATE)

    python -m resource_maximization.mc_dropout_uncertainty --demo   # synthetic data only
"""
from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn

import config

log = logging.getLogger("resource_maximization.mc_dropout_uncertainty")


def enable_mc_dropout(model: nn.Module, rate: float | None = None) -> list[nn.Dropout]:
    """Puts every nn.Dropout submodule into train() mode (so it actually
    samples a mask instead of the eval-mode no-op) while leaving the rest
    of the model's modules untouched. Optionally overrides each dropout
    layer's `p` for this pass (e.g. the plan's "expensive" rate is higher
    than what the model was trained with); the original `p` is stashed on
    the module itself and restored by disable_mc_dropout(), since `p` is a
    plain attribute (not train/eval state) that would otherwise stay
    permanently altered on the shared model object after this call
    returns. Returns the list of dropout modules touched, for callers that
    want to restore eval() mode afterward without calling model.eval()
    (which could be wrong if the caller has other stateful layers
    deliberately left in train mode for an unrelated reason)."""
    dropouts = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    for d in dropouts:
        d.train()
        if rate is not None:
            d._mc_dropout_original_p = d.p
            d.p = rate
    return dropouts


def disable_mc_dropout(dropouts: list[nn.Dropout]) -> None:
    for d in dropouts:
        d.eval()
        original_p = getattr(d, "_mc_dropout_original_p", None)
        if original_p is not None:
            d.p = original_p
            del d._mc_dropout_original_p


def mc_dropout_predict(
    model: nn.Module,
    x: torch.Tensor,
    passes: int,
    rate: float | None = None,
    forward_fn=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Runs `passes` stochastic forward passes with MC-dropout enabled and
    returns (mean_proba, std_proba), each shaped like the model's own
    per-class probability output. `forward_fn` defaults to calling
    `model(x)` and softmaxing its first output (matching
    SSMChallenger.forward's (direction_logits, return_pred) convention);
    pass a custom one for a model whose forward signature differs."""
    if forward_fn is None:
        def forward_fn(m, inp):
            logits, _ = m(inp)
            return torch.softmax(logits, dim=-1)

    was_training = model.training
    dropouts = enable_mc_dropout(model, rate=rate)
    samples = []
    try:
        with torch.no_grad():
            for _ in range(passes):
                samples.append(forward_fn(model, x).cpu().numpy())
    finally:
        disable_mc_dropout(dropouts)
        model.train(was_training)

    stacked = np.stack(samples, axis=0)
    return stacked.mean(axis=0), stacked.std(axis=0)


def cheap_uncertainty(model: nn.Module, x: torch.Tensor, forward_fn=None) -> tuple[np.ndarray, np.ndarray]:
    return mc_dropout_predict(model, x, passes=config.MC_DROPOUT_CHEAP_PASSES,
                               rate=config.MC_DROPOUT_CHEAP_RATE, forward_fn=forward_fn)


def expensive_uncertainty(model: nn.Module, x: torch.Tensor, forward_fn=None) -> tuple[np.ndarray, np.ndarray]:
    return mc_dropout_predict(model, x, passes=config.MC_DROPOUT_EXPENSIVE_PASSES,
                               rate=config.MC_DROPOUT_EXPENSIVE_RATE, forward_fn=forward_fn)


def ensemble_spread(member_probas: list[np.ndarray]) -> np.ndarray:
    """Deep-ensemble disagreement: std across independently-trained
    members' probability outputs (each shaped (batch, n_classes) or
    (n_classes,)). Requires at least 2 members — a single member has no
    ensemble to disagree with, which is a caller error, not a degenerate
    zero-uncertainty case to paper over."""
    if len(member_probas) < 2:
        raise ValueError("ensemble_spread needs at least 2 members to measure disagreement")
    stacked = np.stack(member_probas, axis=0)
    return stacked.std(axis=0)


def tta_predict(
    model: nn.Module,
    x: torch.Tensor,
    n_augmentations: int = 5,
    jitter_std: float = config.TEST_TIME_AUGMENTATION_JITTER_STD,
    forward_fn=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Light test-time augmentation: adds small Gaussian jitter to the
    input features (never to labels/targets — this is inference-only) and
    aggregates. Deliberately reuses `jitter_std` at the plan's own small
    default rather than anything learned, since TTA here is meant as a
    cheap robustness probe, not a data-augmentation training technique."""
    if forward_fn is None:
        def forward_fn(m, inp):
            logits, _ = m(inp)
            return torch.softmax(logits, dim=-1)

    was_training = model.training
    model.eval()
    try:
        samples = []
        with torch.no_grad():
            samples.append(forward_fn(model, x).cpu().numpy())
            for _ in range(n_augmentations - 1):
                jittered = x + torch.randn_like(x) * jitter_std
                samples.append(forward_fn(model, jittered).cpu().numpy())
    finally:
        model.train(was_training)
    stacked = np.stack(samples, axis=0)
    return stacked.mean(axis=0), stacked.std(axis=0)


def combined_uncertainty(
    mc_std: np.ndarray | None = None,
    ensemble_std: np.ndarray | None = None,
    tta_std: np.ndarray | None = None,
) -> float:
    """Aggregates whichever of the three signals a caller has available
    into one scalar (mean over classes/batch of whichever stds were
    provided, then averaged across signal types) for fusion/calibration.py
    to optionally consume downstream. All three are optional since not
    every caller will run every level every tick (the plan's own
    cheap-every-tick / expensive-periodically split)."""
    parts = [s for s in (mc_std, ensemble_std, tta_std) if s is not None]
    if not parts:
        raise ValueError("combined_uncertainty needs at least one uncertainty signal")
    return float(np.mean([float(np.mean(p)) for p in parts]))


def _demo() -> None:
    torch.manual_seed(0)

    class _TinyDropoutModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(5, 16)
            self.dropout = nn.Dropout(0.1)
            self.head = nn.Linear(16, 3)

        def forward(self, x):
            h = torch.relu(self.proj(x))
            h = self.dropout(h)
            return self.head(h), None

    model = _TinyDropoutModel()
    model.eval()
    x = torch.randn(4, 5)

    mean_cheap, std_cheap = cheap_uncertainty(model, x)
    mean_expensive, std_expensive = expensive_uncertainty(model, x)
    print("cheap mean/std shape:", mean_cheap.shape, std_cheap.shape, "mean std:", std_cheap.mean())
    print("expensive mean std:", std_expensive.mean())
    print("model.training after calls (should be False):", model.training)

    member_a = torch.softmax(torch.randn(4, 3), dim=-1).numpy()
    member_b = torch.softmax(torch.randn(4, 3), dim=-1).numpy()
    spread = ensemble_spread([member_a, member_b])
    print("ensemble spread mean:", spread.mean())

    tta_mean, tta_std = tta_predict(model, x)
    print("tta mean std:", tta_std.mean())

    combined = combined_uncertainty(mc_std=std_cheap, ensemble_std=spread, tta_std=tta_std)
    print("combined uncertainty scalar:", combined)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    if args.demo:
        _demo()
    else:
        parser.print_help()
