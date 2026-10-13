"""Offline synthetic test for models/sentiment_branch/train.py's new
--ablation path (sentiment_branch_with vs sentiment_branch_without OOF
export). Monkeypatched load_days/save_oof/purged_walk_forward_splits — no
real Kaggle data, no network calls."""
import shutil
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO_ROOT = Path(r"F:\Downloads\Binance data\feature engineering scripts")
sys.path.insert(0, str(REPO_ROOT))

import config
import common.oof as common_oof
from common.feature_contract import FEATURE_COLUMNS
from models.sentiment_branch import train as sentiment_train

assert set(sentiment_train.SENTIMENT_PROXY_FEATURES).issubset(set(FEATURE_COLUMNS))
assert set(sentiment_train.WITHOUT_SENTIMENT_FEATURES) == set(FEATURE_COLUMNS) - set(sentiment_train.SENTIMENT_PROXY_FEATURES)
assert len(sentiment_train.WITHOUT_SENTIMENT_FEATURES) == len(FEATURE_COLUMNS) - len(sentiment_train.SENTIMENT_PROXY_FEATURES)
print("PASS: WITHOUT_SENTIMENT_FEATURES is exactly FEATURE_COLUMNS minus the sentiment proxies")

n_rows = 400
rng = np.random.default_rng(0)


def _fake_load_days(symbol=config.SYMBOL_UM, max_days=None, prefer_local=True):
    data = {c: rng.normal(size=n_rows).astype(np.float32) for c in FEATURE_COLUMNS}
    data["ts"] = np.arange(n_rows).astype("datetime64[s]").astype("datetime64[us]")
    data["label_triple_barrier"] = rng.integers(-1, 2, size=n_rows).astype(np.int64)
    data["label_is_valid"] = np.ones(n_rows, dtype=bool)
    return pl.DataFrame(data)


saved_oof_calls = []


def _fake_save_oof(member_name, ts, direction_proba):
    saved_oof_calls.append((member_name, len(ts), direction_proba.shape))
    return Path(f"/fake/{member_name}.parquet")


orig_load_days = sentiment_train.load_days
orig_pwfs = sentiment_train.purged_walk_forward_splits
orig_save_oof = common_oof.save_oof
orig_bars_per_day = config.BARS_PER_DAY


def _small_purged_splits(df, n_folds, val_size, purge_bars=5, embargo_bars=5):
    return orig_pwfs(df, n_folds, val_size, purge_bars=5, embargo_bars=5)


sentiment_train.load_days = _fake_load_days
sentiment_train.purged_walk_forward_splits = _small_purged_splits
common_oof.save_oof = _fake_save_oof
config.BARS_PER_DAY = 40  # val_size = val_days * BARS_PER_DAY must stay small relative to n_rows=400

try:
    results = sentiment_train.run_ablation(val_days=1, max_days=None)

    assert set(results.keys()) == {"with", "without"}, results
    assert results["with"]["n_features"] == len(FEATURE_COLUMNS)
    assert results["without"]["n_features"] == len(FEATURE_COLUMNS) - len(sentiment_train.SENTIMENT_PROXY_FEATURES)

    names = {c[0] for c in saved_oof_calls}
    assert names == {"sentiment_branch_with", "sentiment_branch_without"}, names
    for _, n_ts, shape in saved_oof_calls:
        assert shape[1] == 3
        assert n_ts > 0

    model_with = sentiment_train.ABLATION_CHECKPOINT_DIR / "sentiment_branch_with.txt"
    model_without = sentiment_train.ABLATION_CHECKPOINT_DIR / "sentiment_branch_without.txt"
    metrics_path = sentiment_train.ABLATION_CHECKPOINT_DIR / "metrics.json"
    assert model_with.exists() and model_without.exists() and metrics_path.exists()
    print("PASS: run_ablation() produces sentiment_branch_with/without OOF files, checkpoints, and metrics.json")
finally:
    sentiment_train.load_days = orig_load_days
    sentiment_train.purged_walk_forward_splits = orig_pwfs
    common_oof.save_oof = orig_save_oof
    config.BARS_PER_DAY = orig_bars_per_day
    if sentiment_train.ABLATION_CHECKPOINT_DIR.exists():
        shutil.rmtree(sentiment_train.ABLATION_CHECKPOINT_DIR)

print("ALL TESTS PASSED")
