"""
Member-specific online adaptation (§13.1, plan-marked OPTIONAL), gated by
resource_maximization/drift_gate.py's PSI-based should_adapt() decision —
never adapt unconditionally on every tick (the plan's documented
drift-chasing failure mode).

Two adaptation recipes, matching the plan's exact wording:

  GBM family: LightGBM's `refit()` (cheap leaf-value update on new data,
  tree structure fixed) is the default, low-risk adaptation. `continued_boosting()`
  (more rounds via `init_model=`) is reserved for when the drift gate has
  actually fired, since continued boosting alone can drift-chase.

  Deep sequence members (ssm-challenger and friends): `FastWeightsAdapter`
  wraps an existing trained torch model with a small trainable residual
  linear layer ("fast weights") inserted before the model's own head,
  fine-tuned on a drift_gate.build_adaptation_batch() mix of recent data +
  replay-buffer samples. The base model's own parameters are frozen — only
  the adapter's few thousand parameters move — so an adaptation pass can
  never overwrite what full retraining already learned, matching §13.1's
  "never drift arbitrarily far from the last full retrain" anchor
  requirement structurally, not just by convention.

None of this is exercised against a real trained checkpoint anywhere in
this repo yet (per this project's own excluded-from-this-pass Kaggle
training). Every function here is written against the exact checkpoint/
model interfaces models/gbm_family/train.py and models/ssm_challenger/model.py
already define, and is offline-tested against tiny synthetic
boosters/models — never a live Kaggle run.

Usage:
    from resource_maximization.online_adapters import adapt_gbm_member, FastWeightsAdapter
    booster = adapt_gbm_member(booster, X_new, y_new, drift_result)

    python -m resource_maximization.online_adapters --demo   # synthetic data only
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from resource_maximization.drift_gate import should_adapt

if TYPE_CHECKING:
    import torch

log = logging.getLogger("resource_maximization.online_adapters")


# ---------------------------------------------------------------------------
# GBM family adaptation
# ---------------------------------------------------------------------------
def adapt_gbm_member(booster, X_new: np.ndarray, y_new: np.ndarray, drift_result: dict):
    """Applies the plan's §13.1 GBM recipe to an existing lgb.Booster:
    `refit()` unconditionally (cheap, low-risk leaf-value update), and only
    additionally runs a handful of continued-boosting rounds
    (`init_model=booster`) when the drift gate has fired. Returns the
    updated booster; never mutates in place, matching sklearn/LightGBM's
    own convention that fit-like calls return a (possibly new) estimator."""
    import lightgbm as lgb

    if len(X_new) == 0:
        log.info("adapt_gbm_member: no new data — skipping this cycle")
        return booster

    refit_booster = booster.refit(X_new, y_new)

    if not should_adapt(drift_result):
        return refit_booster

    log.info("Drift gate fired (mean_psi=%.4f) — running continued boosting", drift_result["mean_psi"])
    dataset = lgb.Dataset(X_new, label=y_new, free_raw_data=False)
    boosted = lgb.train(
        params={"objective": refit_booster.params.get("objective", "multiclass"),
                "num_class": refit_booster.params.get("num_class"),
                "learning_rate": 0.02, "verbosity": -1},
        train_set=dataset,
        num_boost_round=20,
        init_model=refit_booster,
        keep_training_booster=False,
    )
    return boosted


# ---------------------------------------------------------------------------
# Deep sequence member adaptation
# ---------------------------------------------------------------------------
class FastWeightsAdapter:
    """Wraps a frozen base torch model with a small trainable residual
    adapter layer applied to its pooled hidden state before the base
    model's own heads run. Only `self.adapter`'s parameters are ever
    updated by `fit_step()` — the base model is frozen for the adapter's
    entire lifetime, so adaptation is structurally bounded and can't
    silently overwrite the full-retrain anchor.

    Requires the base model to expose a `d_model`-sized last-hidden-state
    hook; matches models/ssm_challenger/model.SSMChallenger's internal
    shape exactly (input_proj -> blocks -> final_norm gives (batch, seq,
    d_model), pooled to the last timestep) via `hidden_state_fn`, which the
    caller supplies since different sequence members expose this
    differently."""

    def __init__(self, base_model, hidden_state_fn, d_model: int, lr: float = 1e-4):
        import torch
        import torch.nn as nn

        self.torch = torch
        self.base_model = base_model
        for p in self.base_model.parameters():
            p.requires_grad_(False)
        self.hidden_state_fn = hidden_state_fn
        self.adapter = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.adapter.weight)
        nn.init.zeros_(self.adapter.bias)  # starts as a pure identity residual — adaptation begins at "no change"
        self.optimizer = torch.optim.Adam(self.adapter.parameters(), lr=lr)

    def adapted_hidden(self, x) -> "torch.Tensor":
        h = self.hidden_state_fn(self.base_model, x)
        return h + self.adapter(h)

    def fit_step(self, x, y_direction) -> float:
        """One gradient step of the adapter against a batch of
        (x, y_direction) drawn from drift_gate.build_adaptation_batch().
        Returns the scalar loss for logging/monitoring."""
        torch = self.torch
        self.adapter.train()
        h = self.adapted_hidden(x)
        logits = self.base_model.direction_head(h)
        loss = torch.nn.functional.cross_entropy(logits, y_direction)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def predict_proba(self, x) -> np.ndarray:
        torch = self.torch
        self.adapter.eval()
        with torch.no_grad():
            h = self.adapted_hidden(x)
            logits = self.base_model.direction_head(h)
            proba = torch.softmax(logits, dim=-1)
        return proba.cpu().numpy()


def ssm_hidden_state_fn(model, x):
    """hidden_state_fn for models/ssm_challenger/model.SSMChallenger,
    replicating its forward() up to (but not including) direction_head/
    return_head so FastWeightsAdapter can insert its residual in between."""
    h = model.input_proj(x)
    for block in model.blocks:
        h = block(h)
    h = model.dropout(model.final_norm(h))
    return h[:, -1, :]


def _demo() -> None:
    import lightgbm as lgb

    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(500, 5))
    y_train = (X_train[:, 0] > 0).astype(int)
    booster = lgb.train(
        {"objective": "binary", "verbosity": -1},
        lgb.Dataset(X_train, label=y_train),
        num_boost_round=20,
    )
    X_new = rng.normal(size=(100, 5))
    y_new = (X_new[:, 0] > 0).astype(int)

    no_drift = {"mean_psi": 0.01, "per_feature": {}, "n_features_scored": 5}
    drifted = {"mean_psi": 0.5, "per_feature": {}, "n_features_scored": 5}

    refit_only = adapt_gbm_member(booster, X_new, y_new, no_drift)
    boosted = adapt_gbm_member(booster, X_new, y_new, drifted)
    print("refit-only num_trees:", refit_only.num_trees())
    print("continued-boosting num_trees:", boosted.num_trees())

    from models.ssm_challenger.model import SSMChallenger

    torch = __import__("torch")
    model = SSMChallenger(n_features=5, d_model=16, n_layers=1)
    model.eval()
    adapter = FastWeightsAdapter(model, ssm_hidden_state_fn, d_model=16)
    x = torch.randn(8, 10, 5)
    y = torch.randint(0, 3, (8,))
    loss_before = adapter.fit_step(x, y)
    loss_after = adapter.fit_step(x, y)
    print(f"adapter loss: {loss_before:.4f} -> {loss_after:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    if args.demo:
        _demo()
    else:
        parser.print_help()
