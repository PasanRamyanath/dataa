"""
Multi-asset joint training harness (§13.2) for `MultiAssetModel`.
Trains a shared SSM/GRU sequence backbone with per-symbol linear heads across
BTCUSDT and `config.MULTI_ASSET_SYMBOLS`, writing per-symbol OOF predictions
(`multi_asset_btc`, `multi_asset_eth`, ...) and saving a joint checkpoint.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import config
from common.cv import purged_walk_forward_splits
from common.data import FEATURE_COLUMNS, filter_valid, load_days
from common.windowing import make_windows
from resource_maximization.multi_asset_head import MultiAssetModel

log = logging.getLogger("resource_maximization.multi_asset_train")

MEMBER_NAME = "multi-asset"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints" / "multi_asset"
ALL_SYMBOLS: list[str] = [config.SYMBOL_UM, *config.MULTI_ASSET_SYMBOLS]
save_oof = None


def _oof_name_for_symbol(symbol: str) -> str:
    base = symbol.upper()
    if base.endswith("USDT"):
        base = base[:-4]
    return f"multi_asset_{base.lower()}"


def train_and_eval(
    data_by_symbol: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    d_model: int = 128,
    n_layers: int = 4,
    dropout: float = 0.1,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
) -> tuple[MultiAssetModel, dict, dict[str, np.ndarray], dict[str, np.ndarray]]:
    symbols = list(data_by_symbol.keys())
    first_x = data_by_symbol[symbols[0]][0]
    n_features = int(first_x.shape[-1])
    model = MultiAssetModel(
        n_features=n_features,
        symbols=symbols,
        d_model=d_model,
        n_layers=n_layers,
        n_classes=3,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    n_steps = min(max(1, len(data_by_symbol[s][0]) // batch_size) for s in symbols)
    model.train()
    for _epoch in range(epochs):
        for step in range(n_steps):
            optimizer.zero_grad()
            batch_x = {}
            batch_y = {}
            for s in symbols:
                X_tr, y_tr, _, _, _ = data_by_symbol[s]
                start = (step * batch_size) % max(1, len(X_tr) - batch_size + 1)
                xb = torch.as_tensor(X_tr[start : start + batch_size], dtype=torch.float32, device=device)
                yb = torch.as_tensor(y_tr[start : start + batch_size], dtype=torch.long, device=device)
                batch_x[s] = xb
                batch_y[s] = yb
            logits_by_sym = model(batch_x)
            loss = sum(criterion(logits_by_sym[s], batch_y[s]) for s in symbols) / max(1, len(symbols))
            loss.backward()
            optimizer.step()

    model.eval()
    val_proba_by_symbol: dict[str, np.ndarray] = {}
    ts_val_by_symbol: dict[str, np.ndarray] = {}
    per_symbol_metrics: dict[str, dict] = {}

    with torch.no_grad():
        for s in symbols:
            _, _, X_val, y_val, ts_val = data_by_symbol[s]
            xv = torch.as_tensor(X_val, dtype=torch.float32, device=device)
            logits = model({s: xv})[s]
            probs = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float32)
            preds = np.argmax(probs, axis=1)
            acc = float(np.mean(preds == y_val)) if len(y_val) else 0.0
            val_proba_by_symbol[s] = probs
            ts_val_by_symbol[s] = ts_val
            per_symbol_metrics[s] = {
                "n_val_windows": int(len(y_val)),
                "accuracy": acc,
            }

    metrics = {
        "n_symbols": len(symbols),
        "n_steps_per_epoch": int(n_steps),
        "per_symbol": per_symbol_metrics,
    }
    return model, metrics, val_proba_by_symbol, ts_val_by_symbol


def run(
    val_days: int = 7,
    max_days: int | None = None,
    lookback: int = 60,
    stride: int = 10,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    d_model: int = 128,
    n_layers: int = 4,
    dropout: float = 0.1,
    upload: bool = False,
) -> dict:
    from common.oof import save_oof as _default_save_oof
    from common.hf_model_upload import upload_best_checkpoint

    oof_fn = save_oof if save_oof is not None else _default_save_oof
    data_by_symbol = {}
    val_size = val_days * config.BARS_PER_DAY

    for sym in ALL_SYMBOLS:
        df = filter_valid(load_days(symbol=sym, max_days=max_days))
        split = purged_walk_forward_splits(df, n_folds=1, val_size=val_size)[0]
        feat_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
        X_tr, y_tr, _ = make_windows(split.train, lookback=lookback, target_col="label_triple_barrier", feature_cols=feat_cols, stride=stride)
        X_va, y_va, ts_va = make_windows(split.val, lookback=lookback, target_col="label_triple_barrier", feature_cols=feat_cols, stride=stride)
        data_by_symbol[sym] = (X_tr, y_tr.astype(np.int64), X_va, y_va.astype(np.int64), ts_va)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, metrics, val_proba, ts_val = train_and_eval(
        data_by_symbol,
        d_model=d_model,
        n_layers=n_layers,
        dropout=dropout,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = CHECKPOINT_DIR / "multi_asset_model.pt"
    metrics_path = CHECKPOINT_DIR / "metrics.json"
    torch.save(model.state_dict(), model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    for sym, probs in val_proba.items():
        oof_fn(_oof_name_for_symbol(sym), ts_val[sym], probs)

    if upload:
        upload_best_checkpoint(MEMBER_NAME, [model_path, metrics_path], metrics=metrics)

    return metrics
