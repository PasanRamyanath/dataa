"""Offline synthetic test for resource_maximization/multi_asset_train.py.
No live network calls, no real Kaggle data — synthetic per-symbol frames
with monkeypatched load_days/save_oof/upload_best_checkpoint."""
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch

REPO_ROOT = Path(r"F:\Downloads\Binance data\feature engineering scripts")
sys.path.insert(0, str(REPO_ROOT))

import config
from common.feature_contract import FEATURE_COLUMNS
from resource_maximization import multi_asset_train as mat

# --- 1. _oof_name_for_symbol -------------------------------------------------
assert mat._oof_name_for_symbol("BTCUSDT") == "multi_asset_btc"
assert mat._oof_name_for_symbol("ETHUSDT") == "multi_asset_eth"
assert mat._oof_name_for_symbol("SOLUSDT") == "multi_asset_sol"
assert mat._oof_name_for_symbol("BNBUSDT") == "multi_asset_bnb"
print("PASS: _oof_name_for_symbol matches ablation_test.py's documented example (multi_asset_btc)")

# --- 2. train_and_eval() directly on synthetic tensors (core joint-training logic) ---
torch.manual_seed(0)
n_features = 12
symbols = ["BTCUSDT", "ETHUSDT"]

def _synthetic_symbol_data(n_train=64, n_val=32, lookback=20):
    X_train = np.random.randn(n_train, lookback, n_features).astype(np.float32)
    y_train = np.random.randint(0, 3, size=n_train).astype(np.int64)
    X_val = np.random.randn(n_val, lookback, n_features).astype(np.float32)
    y_val = np.random.randint(0, 3, size=n_val).astype(np.int64)
    ts_val = np.arange(n_val).astype("datetime64[s]")
    return X_train, y_train, X_val, y_val, ts_val

data_by_symbol = {s: _synthetic_symbol_data() for s in symbols}

model, metrics, val_proba_by_symbol, ts_val_by_symbol = mat.train_and_eval(
    data_by_symbol, d_model=16, n_layers=1, dropout=0.1,
    epochs=2, batch_size=8, lr=1e-3, device="cpu",
)
assert set(val_proba_by_symbol.keys()) == set(symbols)
for s in symbols:
    assert val_proba_by_symbol[s].shape == (32, 3)
    assert np.allclose(val_proba_by_symbol[s].sum(axis=1), 1.0, atol=1e-4)
    assert metrics["per_symbol"][s]["n_val_windows"] == 32
assert metrics["n_symbols"] == 2
assert metrics["n_steps_per_epoch"] == 64 // 8
print("PASS: train_and_eval() joint-trains a shared backbone and produces per-symbol softmax OOF probabilities")

# Unequal-length symbols: n_steps_per_epoch must be bounded by the SHORTEST loader.
data_by_symbol_uneven = {
    "BTCUSDT": _synthetic_symbol_data(n_train=64),
    "ETHUSDT": _synthetic_symbol_data(n_train=24),
}
_, metrics_uneven, _, _ = mat.train_and_eval(
    data_by_symbol_uneven, d_model=16, n_layers=1, dropout=0.1,
    epochs=1, batch_size=8, lr=1e-3, device="cpu",
)
assert metrics_uneven["n_steps_per_epoch"] == 24 // 8, metrics_uneven
print("PASS: n_steps_per_epoch is correctly bounded by the shortest symbol's loader")

# --- 3. run() end-to-end with monkeypatched load_days/save_oof/upload -------
small_symbols = ["BTCUSDT", "ETHUSDT"]
orig_multi_asset_symbols = config.MULTI_ASSET_SYMBOLS
orig_embargo, orig_purge, orig_bars_per_day = config.EMBARGO_BARS, config.PURGE_BARS, config.BARS_PER_DAY
config.MULTI_ASSET_SYMBOLS = ["ETHUSDT"]
config.EMBARGO_BARS = 5
config.PURGE_BARS = 5
config.BARS_PER_DAY = 40
mat.ALL_SYMBOLS = [config.SYMBOL_UM, *config.MULTI_ASSET_SYMBOLS]

n_rows = 400
lookback = 10

def _fake_load_days(symbol=config.SYMBOL_UM, max_days=None, prefer_local=True):
    rng = np.random.default_rng(abs(hash(symbol)) % (2**32))
    data = {c: rng.normal(size=n_rows).astype(np.float32) for c in FEATURE_COLUMNS}
    data["ts"] = pl.datetime_range(
        start=pl.datetime(2024, 1, 1), end=None, interval="10s", eager=True,
    )[:n_rows] if False else np.arange(n_rows).astype("datetime64[s]")
    data["label_triple_barrier"] = rng.integers(-1, 2, size=n_rows).astype(np.int64)
    data["label_is_valid"] = np.ones(n_rows, dtype=bool)
    return pl.DataFrame(data)

saved_oof_calls = []

def _fake_save_oof(member_name, ts, direction_proba):
    saved_oof_calls.append((member_name, len(ts), direction_proba.shape))
    return Path(f"/fake/{member_name}.parquet")

upload_calls = []

def _fake_upload_best_checkpoint(member_name, checkpoint_paths, metrics=None, api=None):
    upload_calls.append(member_name)
    return f"fake-repo/{member_name}"

import common.data as common_data
import common.oof as common_oof
import common.hf_model_upload as hf_upload

orig_load_days = common_data.load_days
orig_save_oof = common_oof.save_oof
orig_upload = hf_upload.upload_best_checkpoint
mat.load_days = _fake_load_days
mat.save_oof = None  # ensure run() imports save_oof fresh from common.oof (patched below) not a stale ref
common_oof.save_oof = _fake_save_oof
hf_upload.upload_best_checkpoint = _fake_upload_best_checkpoint

try:
    metrics_out = mat.run(
        val_days=1, max_days=None, lookback=lookback, stride=5,
        epochs=1, batch_size=8, lr=1e-3, d_model=8, n_layers=1, dropout=0.1,
        upload=True,
    )
    assert len(saved_oof_calls) == 2, saved_oof_calls
    names = {c[0] for c in saved_oof_calls}
    assert names == {"multi_asset_btc", "multi_asset_eth"}, names
    for _, _, shape in saved_oof_calls:
        assert shape[1] == 3
    assert upload_calls == ["multi-asset"], upload_calls
    model_path = mat.CHECKPOINT_DIR / "multi_asset_model.pt"
    metrics_path = mat.CHECKPOINT_DIR / "metrics.json"
    assert model_path.exists() and metrics_path.exists()
    print("PASS: run() end-to-end produces multi_asset_btc + multi_asset_eth OOF files and a checkpoint")
finally:
    mat.load_days = orig_load_days
    common_oof.save_oof = orig_save_oof
    hf_upload.upload_best_checkpoint = orig_upload
    config.MULTI_ASSET_SYMBOLS = orig_multi_asset_symbols
    config.EMBARGO_BARS, config.PURGE_BARS, config.BARS_PER_DAY = orig_embargo, orig_purge, orig_bars_per_day
    mat.ALL_SYMBOLS = [config.SYMBOL_UM, *config.MULTI_ASSET_SYMBOLS]
    import shutil
    if mat.CHECKPOINT_DIR.exists():
        shutil.rmtree(mat.CHECKPOINT_DIR)

print("ALL TESTS PASSED")
