"""
Naive baselines evaluated through the exact same purged + embargoed
walk-forward CV loop every real model in models/*/train.py uses
(common.cv.purged_walk_forward_splits, itself a thin wrapper around
anti_leakage.py's fold generator).

Nothing here is a model worth deploying — the point is to establish the
floor: if GBM/TFT/PatchTST/etc. can't beat "always predict the majority
class" or "always predict zero return" on the SAME folds, that member isn't
adding anything and shouldn't be in the ensemble. This is the "1" from the
prior recommendation ("wire anti_leakage.py's purged/embargoed folds into an
actual training/eval loop") plus the naive-baseline harness itself.

Baselines:
  classification (label_triple_barrier, {-1,0,1} remapped to {0,1,2}):
    - majority_class: always predict the train fold's most frequent class
    - prior_bar_label: predict the previous bar's own label_triple_barrier
      (a "momentum persistence" naive baseline, not just a constant)
  regression (label_return_h1..h4):
    - zero_return: always predict 0.0 (the standard random-walk /
      efficient-market naive forecast for a log-return series)
    - last_known_return: predict the most recent PAST realized short-horizon
      return (log_return_1, already lagged/trailing in the feature set — see
      common/feature_contract.py) as a naive persistence forecast

Usage:
    python -m validation.naive_baselines --n-folds 5 --val-days 7
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import polars as pl

import config
from common.cv import purged_walk_forward_splits
from common.data import filter_valid, load_days
from common.feature_contract import CLASSIFICATION_TARGET, REGRESSION_TARGETS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("validation.naive_baselines")


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    n = int(mask.sum())
    if n == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "n": 0}
    y_t = y_true[mask]
    y_p = y_pred[mask]
    mae = mean_absolute_error(y_t, y_p)
    rmse = mean_squared_error(y_t, y_p) ** 0.5
    return {"mae": float(mae), "rmse": float(rmse), "n": n}


def evaluate_classification_baselines(train: pl.DataFrame, val: pl.DataFrame) -> dict:
    y_train = (train[CLASSIFICATION_TARGET].to_numpy().astype(np.int64) + 1)
    y_val = (val[CLASSIFICATION_TARGET].to_numpy().astype(np.int64) + 1)

    majority = int(np.bincount(y_train, minlength=3).argmax())
    majority_pred = np.full_like(y_val, majority)

    prior_pred = (val[CLASSIFICATION_TARGET].shift(1).fill_null(0).to_numpy().astype(np.int64) + 1)

    return {
        "majority_class": _classification_metrics(y_val, majority_pred),
        "prior_bar_label": _classification_metrics(y_val, prior_pred),
        "train_class_balance": {str(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))},
        "val_class_balance": {str(k): int(v) for k, v in zip(*np.unique(y_val, return_counts=True))},
    }


def evaluate_regression_baselines(val: pl.DataFrame, target: str) -> dict:
    y_val = val[target].to_numpy().astype(np.float64)
    zero_pred = np.zeros_like(y_val)
    last_known_pred = val["log_return_1"].fill_null(0.0).to_numpy().astype(np.float64)

    return {
        "zero_return": _regression_metrics(y_val, zero_pred),
        "last_known_return": _regression_metrics(y_val, last_known_pred),
    }


def run(n_folds: int, val_days: int, max_days: int | None) -> dict:
    df = filter_valid(load_days(max_days=max_days))

    val_size = val_days * config.BARS_PER_DAY
    splits = purged_walk_forward_splits(df, n_folds=n_folds, val_size=val_size)

    report = {"n_folds": len(splits), "val_size_bars": val_size, "folds": []}
    for split in splits:
        fold_report = {
            "fold_index": split.fold_index,
            "train_rows": split.train.height,
            "val_rows": split.val.height,
            "classification": evaluate_classification_baselines(split.train, split.val),
            "regression": {t: evaluate_regression_baselines(split.val, t) for t in REGRESSION_TARGETS},
        }
        report["folds"].append(fold_report)
        log.info(
            "Fold %d: train=%d val=%d majority_acc=%.4f prior_bar_acc=%.4f",
            split.fold_index, split.train.height, split.val.height,
            fold_report["classification"]["majority_class"]["accuracy"],
            fold_report["classification"]["prior_bar_label"]["accuracy"],
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--val-days", type=int, default=config.VAL_DAYS)
    parser.add_argument("--max-days", type=int, default=None, help="cap total days loaded (dev/debug)")
    parser.add_argument("--out", default=str(config.OUTPUT_ROOT.parent / "validation" / "naive_baselines_report.json"))
    args = parser.parse_args()

    report = run(args.n_folds, args.val_days, args.max_days)

    from pathlib import Path

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("Wrote report to %s", out_path)


if __name__ == "__main__":
    main()
