"""
Offline/synthetic test suite for resource_maximization/* and
validation/ablation_test.py. No real Kaggle checkpoints, no real training,
no live network calls anywhere — every model is a tiny in-memory instance,
every dataframe is synthetic.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import polars as pl
import torch

import config
from resource_maximization.drift_gate import (
    ReplayBuffer, build_adaptation_batch, compute_drift_score, should_adapt,
)
from resource_maximization.online_adapters import FastWeightsAdapter, adapt_gbm_member, ssm_hidden_state_fn
from resource_maximization.multi_asset_head import MultiAssetModel
from resource_maximization.mc_dropout_uncertainty import (
    cheap_uncertainty, combined_uncertainty, disable_mc_dropout, enable_mc_dropout,
    ensemble_spread, expensive_uncertainty, mc_dropout_predict, tta_predict,
)
from validation.ablation_test import compare_variant_to_baseline
from models.ssm_challenger.model import SSMChallenger

PASS = 0
FAIL = 0


def check(name: str, cond: bool) -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"FAIL: {name}")


def test_drift_gate():
    rng = np.random.default_rng(1)
    reference_df = pl.DataFrame({"f1": rng.normal(0, 1, 2000), "f2": rng.normal(5, 2, 2000)})
    stable = pl.DataFrame({"f1": rng.normal(0, 1, 500), "f2": rng.normal(5, 2, 500)})
    drifted = pl.DataFrame({"f1": rng.normal(4, 1, 500), "f2": rng.normal(5, 2, 500)})

    stable_score = compute_drift_score(reference_df, stable, ["f1", "f2"])
    drifted_score = compute_drift_score(reference_df, drifted, ["f1", "f2"])
    check("drift_gate: stable window scores low", stable_score["mean_psi"] < 0.1)
    check("drift_gate: drifted window scores high", drifted_score["mean_psi"] > 0.2)
    check("drift_gate: should_adapt false on stable", not should_adapt(stable_score))
    check("drift_gate: should_adapt true on drifted", should_adapt(drifted_score))

    # NEW regression test for the out-of-range undercounting bug fix
    extreme_recent = pl.DataFrame({
        "f1": np.concatenate([rng.normal(0, 1, 1600), rng.normal(15, 1, 400)]),
        "f2": rng.normal(5, 2, 2000),
    })
    extreme_score = compute_drift_score(reference_df, extreme_recent, ["f1", "f2"])
    check("drift_gate: 20% unprecedented-extreme-value minority is flagged as drift",
          extreme_score["mean_psi"] > 0.1)

    partial_score = compute_drift_score(reference_df, stable.select(["f1"]), ["f1", "f2", "missing_col"])
    check("drift_gate: skips missing/absent columns", partial_score["n_features_scored"] == 1)

    const_ref = pl.DataFrame({"c": np.full(200, 1.0)})
    const_rec = pl.DataFrame({"c": np.full(200, 1.0)})
    const_score = compute_drift_score(const_ref, const_rec, ["c"])
    check("drift_gate: degenerate feature scores 0", const_score["per_feature"]["c"] == 0.0)

    buf = ReplayBuffer(capacity=100, seed=0)
    for i in range(10000):
        buf.add([{"x": i}])
    check("drift_gate: ReplayBuffer respects capacity", len(buf) == 100)
    sampled = buf.sample(50)
    check("drift_gate: ReplayBuffer.sample returns requested count", len(sampled) == 50)
    xs = [row["x"] for row in buf._rows]
    check("drift_gate: ReplayBuffer not just the tail (reservoir, not ring buffer)", min(xs) < 9000)

    recent_df = pl.DataFrame({"x": list(range(20))})
    empty_buf = ReplayBuffer(capacity=50)
    passthrough = build_adaptation_batch(recent_df, empty_buf)
    check("drift_gate: build_adaptation_batch passthrough on empty buffer", passthrough.height == recent_df.height)

    filled_buf = ReplayBuffer(capacity=50, seed=2)
    filled_buf.add_df(pl.DataFrame({"x": list(range(100, 150))}))
    mixed = build_adaptation_batch(recent_df, filled_buf, recent_fraction=0.5)
    check("drift_gate: build_adaptation_batch mixes recent + replay", mixed.height == recent_df.height)
    check("drift_gate: mixed batch contains replay-buffer values", mixed["x"].to_list() != recent_df["x"].to_list())


def test_online_adapters():
    import lightgbm as lgb

    rng = np.random.default_rng(3)
    X_train = rng.normal(size=(500, 5))
    y_train = (X_train[:, 0] > 0).astype(int)
    booster = lgb.train({"objective": "binary", "verbosity": -1}, lgb.Dataset(X_train, label=y_train), num_boost_round=20)

    X_new = rng.normal(size=(100, 5))
    y_new = (X_new[:, 0] > 0).astype(int)

    no_drift = {"mean_psi": 0.01, "per_feature": {}, "n_features_scored": 5}
    drifted = {"mean_psi": 0.5, "per_feature": {}, "n_features_scored": 5}

    refit_only = adapt_gbm_member(booster, X_new, y_new, no_drift)
    check("online_adapters: refit-only keeps tree count", refit_only.num_trees() == booster.num_trees())

    boosted = adapt_gbm_member(booster, X_new, y_new, drifted)
    check("online_adapters: continued boosting adds trees", boosted.num_trees() > booster.num_trees())

    empty_result = adapt_gbm_member(booster, np.empty((0, 5)), np.empty((0,)), drifted)
    check("online_adapters: empty new data is a no-op", empty_result is booster)

    torch.manual_seed(0)
    model = SSMChallenger(n_features=5, d_model=16, n_layers=1)
    model.eval()
    base_params_before = [p.clone() for p in model.parameters()]

    adapter = FastWeightsAdapter(model, ssm_hidden_state_fn, d_model=16)
    adapter_weight_before = adapter.adapter.weight.clone()

    x = torch.randn(8, 10, 5)
    y = torch.randint(0, 3, (8,))
    for _ in range(5):
        adapter.fit_step(x, y)

    base_params_after = list(model.parameters())
    all_frozen = all(torch.equal(a, b) for a, b in zip(base_params_before, base_params_after))
    check("online_adapters: base model params never move", all_frozen)
    check("online_adapters: adapter params DO move", not torch.equal(adapter_weight_before, adapter.adapter.weight))

    proba = adapter.predict_proba(x)
    check("online_adapters: predict_proba shape", proba.shape == (8, 3))
    check("online_adapters: predict_proba sums to 1", np.allclose(proba.sum(axis=1), 1.0, atol=1e-5))


def test_multi_asset_head():
    torch.manual_seed(0)
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    model = MultiAssetModel(n_features=8, symbols=symbols, d_model=16, n_layers=1)
    model.eval()

    batch = {s: torch.randn(4, 12, 8) for s in symbols}
    with torch.no_grad():
        out = model(batch)
    check("multi_asset_head: returns all symbols", set(out.keys()) == set(symbols))
    check("multi_asset_head: correct logits shape", all(v.shape == (4, 3) for v in out.values()))

    with torch.no_grad():
        partial = model({"BTCUSDT": batch["BTCUSDT"]})
    check("multi_asset_head: partial batch handled independently", list(partial.keys()) == ["BTCUSDT"])

    try:
        model({"UNKNOWNUSDT": batch["BTCUSDT"]})
        check("multi_asset_head: unknown symbol raises", False)
    except KeyError:
        check("multi_asset_head: unknown symbol raises", True)

    try:
        MultiAssetModel(n_features=8, symbols=["BTCUSDT", "BTCUSDT"])
        check("multi_asset_head: duplicate symbols raise", False)
    except ValueError:
        check("multi_asset_head: duplicate symbols raise", True)

    shared = model.backbone_state_dict()
    check("multi_asset_head: backbone_state_dict excludes heads", all(not k.startswith("heads.") for k in shared.keys()))
    check("multi_asset_head: backbone_state_dict non-empty", len(shared) > 0)


class _TinyDropoutModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(5, 16)
        self.dropout = torch.nn.Dropout(0.1)
        self.norm = torch.nn.LayerNorm(16)
        self.head = torch.nn.Linear(16, 3)

    def forward(self, x):
        h = torch.relu(self.proj(x))
        h = self.dropout(h)
        h = self.norm(h)
        return self.head(h), None


def test_mc_dropout_uncertainty():
    torch.manual_seed(0)
    model = _TinyDropoutModel()
    model.eval()
    x = torch.randn(6, 5)

    dropouts = enable_mc_dropout(model, rate=0.5)
    check("mc_dropout: dropout put in train mode", all(d.training for d in dropouts))
    check("mc_dropout: non-dropout layers stay in eval mode", not model.norm.training)
    check("mc_dropout: rate override applied", model.dropout.p == 0.5)
    disable_mc_dropout(dropouts)
    check("mc_dropout: dropout restored to eval mode", all(not d.training for d in dropouts))
    check("mc_dropout: original rate restored after disable (regression test)", model.dropout.p == 0.1)

    mean, std = mc_dropout_predict(model, x, passes=15, rate=0.3)
    check("mc_dropout: mean shape correct", mean.shape == (6, 3))
    check("mc_dropout: std shape correct", std.shape == (6, 3))
    check("mc_dropout: model restored to eval after predict", not model.training)
    check("mc_dropout: rate restored after mc_dropout_predict", model.dropout.p == 0.1)
    check("mc_dropout: nonzero variance across passes", std.sum() > 0)

    mean_c, std_c = cheap_uncertainty(model, x)
    mean_e, std_e = expensive_uncertainty(model, x)
    check("mc_dropout: cheap wrapper produces valid shapes", mean_c.shape == (6, 3))
    check("mc_dropout: expensive wrapper produces valid shapes", mean_e.shape == (6, 3))
    check("mc_dropout: rate still restored after both cheap+expensive calls", model.dropout.p == 0.1)

    member_a = np.array([[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]])
    member_b = np.array([[0.6, 0.2, 0.2], [0.1, 0.1, 0.8]])
    spread = ensemble_spread([member_a, member_b])
    check("mc_dropout: ensemble_spread shape", spread.shape == (2, 3))
    check("mc_dropout: ensemble_spread near-zero when members agree", spread[1].sum() < 1e-6)
    try:
        ensemble_spread([member_a])
        check("mc_dropout: ensemble_spread rejects single member", False)
    except ValueError:
        check("mc_dropout: ensemble_spread rejects single member", True)

    model.train()
    tta_mean, tta_std = tta_predict(model, x, n_augmentations=6)
    check("mc_dropout: tta_predict mean shape", tta_mean.shape == (6, 3))
    check("mc_dropout: tta_predict std shape", tta_std.shape == (6, 3))
    check("mc_dropout: tta_predict restores train() mode (regression test)", model.training is True)
    model.eval()

    combined = combined_uncertainty(mc_std=std, ensemble_std=spread, tta_std=tta_std)
    check("mc_dropout: combined_uncertainty returns a scalar", isinstance(combined, float))
    combined_partial = combined_uncertainty(mc_std=std)
    check("mc_dropout: combined_uncertainty works with one signal", isinstance(combined_partial, float))
    try:
        combined_uncertainty()
        check("mc_dropout: combined_uncertainty rejects empty input", False)
    except ValueError:
        check("mc_dropout: combined_uncertainty rejects empty input", True)


def _make_oof(ts, probs):
    return pl.DataFrame({
        "ts": ts,
        "prob_down": probs[:, 0].astype(np.float32),
        "prob_flat": probs[:, 1].astype(np.float32),
        "prob_up": probs[:, 2].astype(np.float32),
    })


def test_ablation_test():
    rng = np.random.default_rng(7)
    n = 900
    ts = np.array(["2026-01-01"], dtype="datetime64[D]").astype("datetime64[ns]") + \
        np.arange(n).astype("timedelta64[h]")
    y_true = rng.integers(0, 3, size=n)

    baseline_pred = (y_true + 1) % 3
    baseline_probs = np.zeros((n, 3))
    baseline_probs[np.arange(n), baseline_pred] = 1.0

    variant_pred = y_true.copy()
    flip_idx = rng.choice(n, size=int(n * 0.1), replace=False)
    variant_pred[flip_idx] = (variant_pred[flip_idx] + 1) % 3
    variant_probs = np.zeros((n, 3))
    variant_probs[np.arange(n), variant_pred] = 1.0

    labels_df = pl.DataFrame({"ts": ts, "label_triple_barrier": (y_true - 1).astype(np.int64)})
    variant_df = _make_oof(ts, variant_probs)
    baseline_df = _make_oof(ts, baseline_probs)

    result = compare_variant_to_baseline(variant_df, baseline_df, labels_df)
    check("ablation_test: evaluated status", result["status"] == "evaluated")
    check("ablation_test: variant with clear lift is adopted", result["decision"] == "adopt_variant")
    check("ablation_test: at least 2 slices passed", result["n_slices_passed"] >= 2)

    identical_result = compare_variant_to_baseline(baseline_df, baseline_df, labels_df)
    check("ablation_test: identical variant/baseline keeps baseline", identical_result["decision"] == "keep_baseline")

    half = n // 2
    first_half_df = _make_oof(ts[:half], variant_probs[:half])
    second_half_df = _make_oof(ts[half:], baseline_probs[half:])
    disjoint_result = compare_variant_to_baseline(first_half_df, second_half_df, labels_df)
    check("ablation_test: disjoint timestamps reported honestly", disjoint_result["status"] == "no_shared_timestamps")

    empty_labels = labels_df.filter(pl.col("ts") < np.datetime64("2000-01-01"))
    no_label_result = compare_variant_to_baseline(variant_df, baseline_df, empty_labels)
    check("ablation_test: no label overlap reported honestly", no_label_result["status"] == "no_overlap_with_labels")


def main():
    test_drift_gate()
    test_online_adapters()
    test_multi_asset_head()
    test_mc_dropout_uncertainty()
    test_ablation_test()
    print(f"\n{PASS} passed, {FAIL} failed")
    if FAIL:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
