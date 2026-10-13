"""Offline synthetic test for resource_maximization/live_glue.py.
No live network calls, no real checkpoints — synthetic members/data with
monkeypatched load_days. Real LiveLoopState() construction is safe here
since no checkpoints exist on disk in this dev environment (every loader
degrades to None, matching production graceful-degradation)."""
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn

REPO_ROOT = Path(r"F:\Downloads\Binance data\feature engineering scripts")
sys.path.insert(0, str(REPO_ROOT))

import config
from common.feature_contract import FEATURE_COLUMNS
from resource_maximization import live_glue

# --- 1. Graceful degradation: fresh state, no checkpoints, no reference snapshot ---
state = live_glue.ResourceMaximizationState(enable_trading=False)
assert state.live.members == {}, "expected no members loaded (no checkpoints in this dev env)"
report = live_glue.maybe_adapt_gbm(state)
assert report == {"status": "no_reference_snapshot"}, report
uncertainty = live_glue.maybe_estimate_uncertainty(state)
assert uncertainty is None, uncertainty
print("PASS: graceful degradation with no checkpoints/reference snapshot present")

# --- 2. maybe_estimate_uncertainty with a fake ssm-challenger member ---
orig_lookback = config.LOOKBACK_PERIODS
config.LOOKBACK_PERIODS = 12
n_features = 10
n_rows = 20


class _TinyDropoutModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(n_features, 8)
        self.dropout = nn.Dropout(0.2)
        self.head = nn.Linear(8, 3)

    def forward(self, x):
        h = torch.relu(self.proj(x[:, -1, :]))
        h = self.dropout(h)
        return self.head(h), None


class _FakeSsmMember:
    name = "ssm-challenger"

    def __init__(self, model):
        self.model = model
        self.torch = torch
        self.device = "cpu"


try:
    feat_data = {c: np.random.randn(n_rows).astype(np.float32) for c in FEATURE_COLUMNS[:n_features]}
    extra_cols = {c: np.zeros(n_rows, dtype=np.float32) for c in FEATURE_COLUMNS[n_features:]}
    df = pl.DataFrame({**feat_data, **extra_cols, "close": np.full(n_rows, 100.0), "ts": np.arange(n_rows).astype("datetime64[s]").astype("datetime64[us]")})
    state.live.buffer.df = df
    state.live.members["ssm-challenger"] = _FakeSsmMember(_TinyDropoutModel().eval())

    result = live_glue.maybe_estimate_uncertainty(state)
    assert result is not None
    assert len(result["mean_proba"]) == 3
    assert len(result["std_proba"]) == 3
    assert result["mean_uncertainty"] >= 0.0
    print("PASS: maybe_estimate_uncertainty produces MC-dropout mean/std over a fake ssm-challenger member")

    # Buffer too short -> gracefully returns None, not a crash.
    state.live.buffer.df = df.head(3)
    assert live_glue.maybe_estimate_uncertainty(state) is None
    print("PASS: maybe_estimate_uncertainty returns None when the buffer is shorter than LOOKBACK_PERIODS")
finally:
    config.LOOKBACK_PERIODS = orig_lookback

# --- 3. maybe_adapt_gbm end-to-end: real drift_gate + online_adapters against a real tiny lightgbm booster ---
import lightgbm as lgb

rng = np.random.default_rng(0)
n_ref, n_recent = 300, 150

def _synthetic_feature_frame(n_rows, shift: float):
    data = {c: rng.normal(loc=shift, size=n_rows).astype(np.float32) for c in FEATURE_COLUMNS}
    data["ts"] = np.arange(n_rows).astype("datetime64[s]").astype("datetime64[us]")
    data["label_triple_barrier"] = rng.integers(-1, 2, size=n_rows).astype(np.int64)
    data["label_is_valid"] = np.ones(n_rows, dtype=bool)
    return pl.DataFrame(data)

reference_df = _synthetic_feature_frame(n_ref, shift=0.0)
drifted_recent_df = _synthetic_feature_frame(n_recent, shift=3.0)  # large shift -> should trigger adaptation
stable_recent_df = _synthetic_feature_frame(n_recent, shift=0.0)   # no shift -> refit only, no continued boosting

X_train = reference_df.select(FEATURE_COLUMNS).to_numpy().astype(np.float32)
y_train = (reference_df["label_triple_barrier"].to_numpy() + 1).astype(np.int64)
booster = lgb.train(
    {"objective": "multiclass", "num_class": 3, "verbosity": -1},
    lgb.Dataset(X_train, label=y_train),
    num_boost_round=10,
)


class _FakeGbmMember:
    name = "gbm-family"

    def __init__(self, booster):
        self.booster = booster


state2 = live_glue.ResourceMaximizationState(enable_trading=False)
state2.reference_df = reference_df
state2.live.members["gbm-family"] = _FakeGbmMember(booster)

orig_load_days = live_glue.load_days
try:
    live_glue.load_days = lambda max_days=None: drifted_recent_df
    report = live_glue.maybe_adapt_gbm(state2)
    assert report["status"] == "checked", report
    assert report["adapted"] is True, report
    assert report["drift"]["mean_psi"] >= config.DRIFT_PSI_THRESHOLD, report
    assert len(state2.replay_buffer) == n_recent
    print("PASS: maybe_adapt_gbm detects real drift and adapts the booster (refit + continued boosting)")

    state3 = live_glue.ResourceMaximizationState(enable_trading=False)
    state3.reference_df = reference_df
    state3.live.members["gbm-family"] = _FakeGbmMember(booster)
    live_glue.load_days = lambda max_days=None: stable_recent_df
    report_stable = live_glue.maybe_adapt_gbm(state3)
    assert report_stable["status"] == "checked", report_stable
    assert report_stable["adapted"] is False, report_stable
    assert report_stable["drift"]["mean_psi"] < config.DRIFT_PSI_THRESHOLD, report_stable
    print("PASS: maybe_adapt_gbm leaves the booster's adaptation flag False when there's no real drift")

    # Cadence: only ticks where tick_count % N == 0 actually check.
    state4 = live_glue.ResourceMaximizationState(enable_trading=False)
    state4.reference_df = reference_df
    state4.live.members["gbm-family"] = _FakeGbmMember(booster)
    state4.tick_count = 1
    assert live_glue.maybe_adapt_gbm(state4) is None
    print("PASS: maybe_adapt_gbm respects LIVE_GLUE_DRIFT_CHECK_EVERY_N_TICKS cadence")
finally:
    live_glue.load_days = orig_load_days

# --- 4. run_glued_tick folds everything into one record, enable_adaptation=False skips GBM only ---
orig_run_tick = live_glue.live_loop.run_tick
live_glue.live_loop.run_tick = lambda live_state: {"status": "scored", "ts": "2024-01-01"}
try:
    state5 = live_glue.ResourceMaximizationState(enable_trading=False)
    state5.reference_df = reference_df
    state5.live.members["gbm-family"] = _FakeGbmMember(booster)
    live_glue.load_days = lambda max_days=None: stable_recent_df

    record_with = live_glue.run_glued_tick(state5, enable_adaptation=True)
    assert record_with["resource_maximization"]["gbm_adaptation"] is not None
    assert state5.tick_count == 1

    record_without = live_glue.run_glued_tick(state5, enable_adaptation=False)
    assert record_without["resource_maximization"]["gbm_adaptation"] is None
    assert state5.tick_count == 2
    print("PASS: run_glued_tick folds gbm_adaptation/mc_dropout_uncertainty into one record and enable_adaptation=False skips only GBM")
finally:
    live_glue.live_loop.run_tick = orig_run_tick
    live_glue.load_days = orig_load_days

print("ALL TESTS PASSED")
