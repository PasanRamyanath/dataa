"""
Fixed-length lookback windowing for sequence models that consume raw
tensors directly rather than going through neuralforecast (models/
ssm_challenger/, models/kronos_finetune/) — TFT/PatchTST/iTransformer/NHITS
do their own windowing internally via neuralforecast's input_size/h, so they
don't use this module.

Window i covers rows [i-lookback+1, i] (inclusive, strictly at-or-before i —
no leakage) and is labeled with row i's own label columns, which are
themselves already forward-looking by construction (triple-barrier,
forward-return horizons — see labeling.py); that's the label's intended
semantics, not a leak introduced by windowing.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from common.feature_contract import FEATURE_COLUMNS


def make_windows(
    df: pl.DataFrame,
    lookback: int,
    target_col: str,
    feature_cols: list[str] | None = None,
    stride: int = 1,
    remap_triple_barrier: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X, y, ts) where X has shape (n_windows, lookback, n_features).

    `stride` subsamples window START positions (every `stride`-th valid
    window) — at 10s bars a lookback of 720 with stride=1 produces one
    window per bar, which is far more than needed for a single epoch to see
    every distinct pattern once; a larger stride trades window density for
    training throughput without changing what any individual window looks
    like."""
    feature_cols = feature_cols or FEATURE_COLUMNS
    X_full = df.select(feature_cols).with_columns(
        [pl.col(c).cast(pl.Float32).fill_null(0.0) for c in feature_cols]
    ).to_numpy().astype(np.float32)

    y_raw = df[target_col].to_numpy()
    if target_col == "label_triple_barrier" and remap_triple_barrier:
        y_full = y_raw.astype(np.int64) + 1
    elif target_col == "label_meta_act":
        y_full = y_raw.astype(np.int64)
    else:
        y_full = y_raw.astype(np.float32)

    ts_full = df["ts"].to_numpy()

    n = X_full.shape[0]
    if n < lookback:
        raise ValueError(f"make_windows: df has {n} rows, need at least lookback={lookback}")

    end_indices = np.arange(lookback - 1, n, stride)
    X = np.stack([X_full[i - lookback + 1: i + 1] for i in end_indices])
    y = y_full[end_indices]
    ts = ts_full[end_indices]
    return X, y, ts


def run_ids(df: pl.DataFrame, interval_seconds: int, ts_col: str = "ts") -> np.ndarray:
    """Per-row id of the contiguous run (see common.cv.contiguous_runs) the
    row belongs to — increments at every gap between consecutive rows."""
    ts = df[ts_col]
    if isinstance(ts.dtype, pl.Datetime):
        ts = ts.dt.cast_time_unit("ns")
    ts_i64 = ts.cast(pl.Int64).to_numpy()
    if len(ts_i64) == 0:
        return np.zeros(0, dtype=np.int64)
    breaks = np.diff(ts_i64) != interval_seconds * 1_000_000_000
    return np.concatenate([[0], np.cumsum(breaks)]).astype(np.int64)


def window_end_indices(
    df: pl.DataFrame,
    lookback: int,
    interval_seconds: int,
    start_row: int = 0,
    end_row: int | None = None,
    stride: int = 1,
    horizon: int = 0,
    valid_col: str | None = None,
    nonnull_cols: tuple[str, ...] = (),
) -> np.ndarray:
    """Row indices i in [start_row, end_row) usable as the LAST context row of
    a window over the unfiltered, gap-containing output of common.data.
    load_days: rows [i-lookback+1, i+horizon] must all lie in one contiguous
    run (so no window ever straddles a missing day), row i must pass
    `valid_col` (e.g. label_is_valid — labels there are placeholders, see
    labeling.py) and have non-null `nonnull_cols`. Context rows before
    start_row may be used (e.g. a val window's lookback reaching into the
    embargo zone — features only, never labels). `stride` subsamples the
    survivors."""
    n = df.height
    end_row = n if end_row is None else min(end_row, n)
    rid = run_ids(df, interval_seconds)
    idx = np.arange(max(start_row, lookback - 1), max(end_row - horizon, 0))
    if len(idx) == 0:
        return idx
    ok = (rid[idx - lookback + 1] == rid[idx]) & (rid[idx + horizon] == rid[idx])
    if valid_col is not None:
        ok &= df[valid_col].fill_null(False).to_numpy()[idx]
    for c in nonnull_cols:
        ok &= df[c].is_not_null().to_numpy()[idx]
    return idx[ok][::stride]


def feature_matrix(df: pl.DataFrame, feature_cols: list[str] | None = None) -> np.ndarray:
    """(n_rows, n_features) float32, nulls -> 0 — the per-row matrix a
    LazyWindowDataset slices windows out of (built once, never stacked)."""
    feature_cols = feature_cols or FEATURE_COLUMNS
    return df.select([pl.col(c).cast(pl.Float32).fill_null(0.0) for c in feature_cols]).to_numpy().astype(np.float32, copy=False)


SCALER_CLIP = 10.0


def fit_feature_scaler(features: np.ndarray, fit_rows: int) -> dict[str, np.ndarray]:
    """Per-feature mean/std over rows [0, fit_rows) only (the training
    window — never val). Neural members need it: raw columns span ~1e-5
    (returns) to ~1e10 (open_interest_value), which a Linear input layer
    can't learn across. Stored in the checkpoint so inference applies the
    identical transform."""
    rows = features[:fit_rows].astype(np.float64)
    mean = rows.mean(axis=0)
    std = rows.std(axis=0)
    scale = np.where(std > 1e-12, std, 1.0)
    return {"mean": mean.astype(np.float32), "scale": scale.astype(np.float32)}


def apply_feature_scaler(features: np.ndarray, scaler: dict[str, np.ndarray] | None) -> np.ndarray:
    """(x - mean) / scale, clipped to +-SCALER_CLIP (heavy-tailed crypto
    volume/liquidation spikes would otherwise dominate a batch). Identity
    when `scaler` is None (a checkpoint from before scaling existed)."""
    if scaler is None:
        return features
    out = (features - scaler["mean"]) / scaler["scale"]
    return np.clip(out, -SCALER_CLIP, SCALER_CLIP).astype(np.float32, copy=False)


class LazyWindowDataset:
    """torch-compatible map-style dataset yielding (X[i-lookback+1:i+1],
    *targets[i]) per end index — slices on access instead of materializing
    every (lookback, n_features) window up front like make_windows does
    (which costs ~0.5GB per training day at lookback 720 / stride 10 and
    OOMs a full-history run). `features` and `targets` are shared, not
    copied, across datasets built on the same frame."""

    def __init__(self, features: np.ndarray, end_indices: np.ndarray, lookback: int, *targets: np.ndarray):
        import torch

        self.features = torch.from_numpy(features)
        self.ends = np.asarray(end_indices, dtype=np.int64)
        self.lookback = lookback
        self.targets = [torch.from_numpy(t) for t in targets]

    def __len__(self) -> int:
        return len(self.ends)

    def __getitem__(self, k: int):
        i = int(self.ends[k])
        return (self.features[i - self.lookback + 1: i + 1], *(t[i] for t in self.targets))
