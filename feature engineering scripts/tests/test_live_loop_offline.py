"""Offline test of inference.live_loop — no live network, no real checkpoints
except a temporary synthetic gbm-family one written into the real repo's
models/gbm_family/checkpoints/ (confirmed absent before, removed after), to
exercise both the zero-members graceful-degradation path and the
one-member-voting path end to end."""
import json
import os
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO = str(Path(__file__).resolve().parent.parent)
SCRATCH = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(SCRATCH, "live_loop_smoke", "output")
PAPER_ROOT = os.path.join(SCRATCH, "live_loop_smoke", "paper_trading")
TICKLOG_ROOT = os.path.join(SCRATCH, "live_loop_smoke", "tick_log")
shutil.rmtree(os.path.join(SCRATCH, "live_loop_smoke"), ignore_errors=True)
os.makedirs(OUTPUT_ROOT, exist_ok=True)

os.environ["FEATURE_OUTPUT_ROOT"] = OUTPUT_ROOT
os.environ["PAPER_TRADING_DATA_ROOT"] = PAPER_ROOT
os.environ["LIVE_TICK_LOG_ROOT"] = TICKLOG_ROOT

sys.path.insert(0, REPO)

import numpy as np
import polars as pl

import config
from common.feature_contract import FEATURE_COLUMNS, LABEL_COLUMNS, CATEGORICAL_OR_FLAG_FEATURE_COLUMNS

assert str(config.OUTPUT_ROOT) == OUTPUT_ROOT, f"config.OUTPUT_ROOT did not pick up env override: {config.OUTPUT_ROOT}"

rng = np.random.default_rng(42)
N = 60  # small — enough for the GBM member (needs only 1 row) but well short of ssm-challenger's 720

def make_synthetic_day(date_str: str, n: int) -> pl.DataFrame:
    start = datetime.fromisoformat(date_str)
    ts = [start + timedelta(seconds=10 * i) for i in range(n)]
    data = {"ts": ts}
    price = 60000.0 + np.cumsum(rng.normal(0, 5, n))
    data["open"] = price.astype(np.float32)
    data["high"] = (price + rng.uniform(0, 10, n)).astype(np.float32)
    data["low"] = (price - rng.uniform(0, 10, n)).astype(np.float32)
    data["close"] = (price + rng.normal(0, 2, n)).astype(np.float32)
    data["volume"] = rng.uniform(1, 100, n).astype(np.float32)
    for col in FEATURE_COLUMNS:
        if col in data:
            continue
        if col in CATEGORICAL_OR_FLAG_FEATURE_COLUMNS:
            data[col] = rng.integers(0, 2, n).astype(np.int32)
        else:
            data[col] = rng.normal(0, 1, n).astype(np.float32)
    for col in LABEL_COLUMNS:
        if col == "label_triple_barrier":
            data[col] = rng.integers(-1, 2, n).astype(np.int8)
        elif col == "label_barrier_type":
            data[col] = ["time"] * n
        elif col == "label_is_valid":
            data[col] = np.array([True] * n)
        elif col == "label_meta_act":
            data[col] = rng.integers(0, 2, n).astype(np.int8)
        else:
            data[col] = rng.normal(0, 0.001, n).astype(np.float32)
    return pl.DataFrame(data)

df1 = make_synthetic_day("2026-01-01T00:00:00", N)
df1.write_parquet(os.path.join(OUTPUT_ROOT, f"features_{config.SYMBOL_UM}_2026-01-01.parquet"))
print("Wrote synthetic day 1:", df1.height, "rows")

from inference import live_loop

# ---- Test A: zero members loaded (no checkpoints anywhere) -> graceful no-op fusion ----
state = live_loop.LiveLoopState(buffer_bars=live_loop.DEFAULT_BUFFER_BARS, enable_trading=True)
assert len(state.members) == 0, f"expected 0 members loaded, got {list(state.members)}"
record_a = live_loop.run_tick(state)
print("Test A tick record:", record_a)
assert record_a["status"] == "scored"
assert record_a["n_members_voted"] == 0
assert record_a["prob_down"] is None
assert "trading" not in record_a, "should not place a trade with no primary_direction"
print("TEST A PASSED: zero-member graceful degradation works")

# second tick with no new data -> no_new_bar
record_a2 = live_loop.run_tick(state)
assert record_a2["status"] == "no_new_bar", f"expected no_new_bar, got {record_a2}"
print("TEST A2 PASSED: repeat tick with no new bar correctly skipped")

# ---- Test B: train + drop a real gbm-family checkpoint, verify one member votes and trades ----
gbm_ckpt_dir = Path(REPO) / "models" / "gbm_family" / "checkpoints"
assert not gbm_ckpt_dir.exists(), f"expected no pre-existing gbm_family checkpoints at {gbm_ckpt_dir}, found some — aborting to avoid clobbering real work"
gbm_ckpt_dir.mkdir(parents=True, exist_ok=True)
try:
    import lightgbm as lgb

    X_train = rng.normal(0, 1, (200, len(FEATURE_COLUMNS))).astype(np.float32)
    y_train = rng.integers(0, 3, 200)
    train_set = lgb.Dataset(X_train, label=y_train)
    booster = lgb.train({"objective": "multiclass", "num_class": 3, "verbosity": -1}, train_set, num_boost_round=5)
    booster.save_model(str(gbm_ckpt_dir / "direction_classifier.txt"))
    print("Wrote synthetic gbm-family checkpoint")

    state_b = live_loop.LiveLoopState(buffer_bars=live_loop.DEFAULT_BUFFER_BARS, enable_trading=True)
    assert list(state_b.members) == ["gbm-family"], f"expected only gbm-family loaded, got {list(state_b.members)}"

    # Write a second day so the buffer sees "new" data relative to a fresh state
    df2 = make_synthetic_day("2026-01-02T00:00:00", N)
    df2.write_parquet(os.path.join(OUTPUT_ROOT, f"features_{config.SYMBOL_UM}_2026-01-02.parquet"))

    record_b = live_loop.run_tick(state_b)
    print("Test B tick record:", record_b)
    assert record_b["status"] == "scored"
    assert record_b["n_members_voted"] == 1
    assert record_b["members_voted"] == ["gbm-family"]
    assert record_b["fusion_stage"] == "mean_vote_no_stacking_checkpoint"
    assert record_b["prob_down"] is not None and record_b["prob_up"] is not None
    proba_sum = record_b["prob_down"] + record_b["prob_flat"] + record_b["prob_up"]
    assert abs(proba_sum - 1.0) < 1e-4, f"fused probabilities should sum to 1, got {proba_sum}"
    # No meta-labeling filter loaded -> no act_probability/bet_size -> engine must not place a trade
    assert record_b["act_probability"] is None
    assert "trading" not in record_b, "engine should not trade without a meta-labeling bet_size"
    print("TEST B PASSED: single-member mean-vote fusion works, no trade without meta-labeling gate")

    # Verify tick log + engine state files were actually written
    tick_log_files = list(Path(TICKLOG_ROOT).glob("*.jsonl"))
    assert len(tick_log_files) == 1, f"expected 1 tick log file, got {tick_log_files}"
    lines = tick_log_files[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2, f"expected 2 logged ticks (A and B; A2 no_new_bar is not logged), got {len(lines)}"
finally:
    shutil.rmtree(gbm_ckpt_dir, ignore_errors=True)
    print("Cleaned up temporary gbm-family checkpoint from real repo")

print("ALL LIVE_LOOP OFFLINE TESTS PASSED")
