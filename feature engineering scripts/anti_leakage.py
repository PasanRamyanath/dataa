"""
Anti-leakage utilities: purged + embargoed walk-forward split generation,
and an empirical leakage self-test.

No prior code in this project actually implemented purged walk-forward CV
(only a simple last-N-days holdout existed in train_tft_kaggle.py, and only
prose describing the proper scheme existed in the planning docs) — this
module is a from-scratch implementation of that design.
"""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from config import EMBARGO_BARS, PURGE_BARS


@dataclass
class Fold:
    train_end: int      # exclusive
    embargo_end: int     # exclusive; [train_end, embargo_end) is discarded (== val_start)
    val_start: int
    val_end: int         # exclusive
    purge_start: int     # start of the purge sub-zone within [train_end, embargo_end); [train_end, purge_start) is the embargo sub-zone


def make_expanding_walk_forward_folds(
    n_rows: int,
    n_folds: int,
    val_size: int,
    purge_bars: int = PURGE_BARS,
    embargo_bars: int = EMBARGO_BARS,
) -> list[Fold]:
    """Expanding-window walk-forward folds with a purge gap (removed from the
    END of the training window, since labels near the boundary can still
    reference outcomes inside the validation window) and an embargo gap
    (removed just before that, so validation features' own rolling windows
    can't reach back into training data). Both gaps are carved out of the
    training window's tail — val_start/val_end are never shifted, so
    consecutive folds' validation windows stay contiguous — giving a combined
    discarded zone [train_end, val_start) of width purge_bars + embargo_bars,
    not just purge_bars."""
    folds: list[Fold] = []
    val_start_min = n_rows - n_folds * val_size
    if val_start_min <= purge_bars + embargo_bars:
        raise ValueError(
            f"n_rows={n_rows} too small for {n_folds} folds of val_size={val_size} "
            f"with purge={purge_bars}/embargo={embargo_bars}."
        )
    for i in range(n_folds):
        val_start = val_start_min + i * val_size
        val_end = val_start + val_size
        purge_start = max(0, val_start - purge_bars)
        train_end = max(0, purge_start - embargo_bars)
        embargo_end = val_start
        folds.append(Fold(train_end=train_end, embargo_end=embargo_end, val_start=val_start,
                           val_end=val_end, purge_start=purge_start))
    return folds


def split_train_val(df: pl.DataFrame, fold: Fold) -> tuple[pl.DataFrame, pl.DataFrame]:
    train = df.slice(0, fold.train_end)
    val = df.slice(fold.val_start, fold.val_end - fold.val_start)
    return train, val


def assert_no_split_overlap(train: pl.DataFrame, val: pl.DataFrame, ts_col: str = "ts") -> None:
    train_max_ts = train.select(pl.col(ts_col).max()).item()
    val_min_ts = val.select(pl.col(ts_col).min()).item()
    if train_max_ts is not None and val_min_ts is not None and train_max_ts >= val_min_ts:
        raise AssertionError(
            f"Train/val split overlaps in time: train max ts={train_max_ts} >= val min ts={val_min_ts}"
        )


def empirical_leakage_check(
    df: pl.DataFrame,
    feature_col: str,
    recompute_fn,
    n_samples: int = 20,
    rtol: float = 1e-4,
    seed: int = 42,
) -> dict:
    """Recompute `feature_col` at a handful of randomly sampled cut points
    using ONLY the data up to and including that point, and compare against
    the value the full-dataframe computation produced at the same row.

    A mismatch beyond `rtol` is strong evidence the feature's rolling window
    (or the pipeline stage that built it) reached forward past its own
    timestamp. `recompute_fn(truncated_df) -> pl.Series` must return the
    same feature computed on a dataframe truncated to just that cut point.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    n = df.height
    if n < 100:
        return {"checked": 0, "mismatches": [], "ok": True}

    lo = n // 4  # avoid the warm-up region where rolling windows are still filling
    candidates = rng.choice(np.arange(lo, n), size=min(n_samples, n - lo), replace=False)

    full_values = df[feature_col].to_numpy()
    mismatches = []
    for idx in candidates:
        idx = int(idx)
        truncated = df.slice(0, idx + 1)
        recomputed = recompute_fn(truncated)
        recomputed_last = recomputed[-1]
        full_val = full_values[idx]
        if full_val is None or recomputed_last is None:
            continue
        if not np.isclose(full_val, recomputed_last, rtol=rtol, equal_nan=True):
            mismatches.append({"row": idx, "full": float(full_val), "recomputed": float(recomputed_last)})

    return {"checked": len(candidates), "mismatches": mismatches, "ok": len(mismatches) == 0}
