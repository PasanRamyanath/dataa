"""
Thin, reusable wrapper around anti_leakage.py's purged + embargoed
walk-forward fold generator, so every training script (validation baselines,
GBM, sequence models, HMM, fusion) shares the exact same split logic instead
of each re-deriving its own train/val boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from anti_leakage import Fold, assert_no_split_overlap, make_expanding_walk_forward_folds, split_train_val
from config import EMBARGO_BARS, PURGE_BARS


@dataclass
class CVSplit:
    fold_index: int
    fold: Fold
    train: pl.DataFrame
    val: pl.DataFrame


def purged_walk_forward_splits(
    df: pl.DataFrame,
    n_folds: int,
    val_size: int,
    purge_bars: int = PURGE_BARS,
    embargo_bars: int = EMBARGO_BARS,
) -> list[CVSplit]:
    """Builds and materializes `n_folds` expanding-window purged/embargoed
    splits over an already-sorted-by-ts `df`. Asserts no train/val time
    overlap on every fold before returning (fails fast rather than letting a
    silently-leaked split reach a model)."""
    folds = make_expanding_walk_forward_folds(df.height, n_folds, val_size, purge_bars, embargo_bars)
    splits = []
    for i, fold in enumerate(folds):
        train, val = split_train_val(df, fold)
        assert_no_split_overlap(train, val)
        splits.append(CVSplit(fold_index=i, fold=fold, train=train, val=val))
    return splits


def _ts_to_seconds(df: pl.DataFrame, ts_col: str = "ts") -> np.ndarray:
    ts_arr = df[ts_col].to_numpy()
    if len(ts_arr) == 0:
        return np.zeros(0, dtype=np.int64)
    if np.issubdtype(ts_arr.dtype, np.datetime64):
        return ts_arr.astype("datetime64[s]").astype(np.int64)
    if ts_arr.dtype == object and hasattr(ts_arr[0], "timestamp"):
        return np.array([int(x.timestamp()) for x in ts_arr], dtype=np.int64)
    ts_int = ts_arr.astype(np.int64)
    if len(ts_int) > 1:
        diffs = np.abs(np.diff(ts_int[:min(100, len(ts_int))]))
        nz = diffs[diffs > 0]
        med_diff = int(np.median(nz)) if len(nz) > 0 else 0
        if med_diff >= 1_000_000_000:
            return ts_int // 1_000_000_000
        elif med_diff >= 1_000_000:
            return ts_int // 1_000_000
        elif med_diff >= 1_000:
            return ts_int // 1_000
    return ts_int


def assert_contiguous(df: pl.DataFrame, interval_seconds: int, ts_col: str = "ts") -> None:
    """Sequence models (TFT/PatchTST/iTransformer/N-HiTS/SSM/Kronos) assume a
    fixed-step bar grid with no missing bars inside a window."""
    ts_sec = _ts_to_seconds(df, ts_col)
    if len(ts_sec) <= 1:
        return
    bad = np.sum(np.diff(ts_sec) != interval_seconds)
    if bad > 0:
        raise AssertionError(
            f"{int(bad)} non-uniform gap(s) found in '{ts_col}' (expected {interval_seconds}s steps); "
            "sequence models must window within contiguous runs only — see common.cv.contiguous_runs"
        )


def contiguous_runs(df: pl.DataFrame, interval_seconds: int, ts_col: str = "ts") -> list[pl.DataFrame]:
    """Splits `df` into maximal runs of uniformly-spaced rows, at every gap."""
    ts_sec = _ts_to_seconds(df, ts_col)
    if len(ts_sec) == 0:
        return []
    gap_idx = np.where(np.diff(ts_sec) != interval_seconds)[0] + 1
    bounds = [0, *gap_idx.tolist(), len(ts_sec)]
    return [df.slice(bounds[i], bounds[i + 1] - bounds[i]) for i in range(len(bounds) - 1)]
