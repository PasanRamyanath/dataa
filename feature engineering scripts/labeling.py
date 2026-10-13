"""
Label construction: volatility-scaled triple-barrier labels, multi-horizon
forward returns, and a meta-label (act / avoid).

Unlike every module in feature_engine/, this module is EXPECTED to look
forward in time — that is what a label is. The anti-leakage rule this
pipeline enforces is that labels must never be fed back in as features, and
that train/val/test splits must be purged/embargoed around label horizons
(see anti_leakage.py) so a training window's labels never overlap a
validation window's feature inputs.

Triple-barrier method (Lopez de Prado): for each bar t, set an upper
(profit-take) and lower (stop-loss) barrier scaled by the bar's own trailing
realized volatility (known at t, so barrier width itself is leakage-free),
then scan forward up to `max_hold` bars for whichever barrier is touched
first; if neither is touched, the outcome is decided by a time barrier
(sign of the return at max_hold).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from config import (
    TRIPLE_BARRIER_MAX_HOLD_BARS,
    TRIPLE_BARRIER_PT_MULT,
    TRIPLE_BARRIER_SL_MULT,
    TRIPLE_BARRIER_VOL_LOOKBACK,
    LABEL_HORIZONS_BARS,
    PRIMARY_INTERVAL_SECONDS,
)


def _first_touch_index(breach: np.ndarray) -> np.ndarray:
    """breach: (n, max_hold) bool matrix -> index of first True per row, or -1."""
    any_touch = breach.any(axis=1)
    first_idx = np.argmax(breach, axis=1)
    return np.where(any_touch, first_idx, -1)


def triple_barrier_labels(
    df: pl.DataFrame,
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    max_hold: int = TRIPLE_BARRIER_MAX_HOLD_BARS,
    pt_mult: float = TRIPLE_BARRIER_PT_MULT,
    sl_mult: float = TRIPLE_BARRIER_SL_MULT,
    vol_lookback: int = TRIPLE_BARRIER_VOL_LOOKBACK,
) -> pl.DataFrame:
    n = df.height
    close = df[close_col].to_numpy()
    high = df[high_col].to_numpy()
    low = df[low_col].to_numpy()

    log_close = np.log(close)
    ret1 = np.diff(log_close, prepend=log_close[0])
    # Per-bar sigma from a trailing rolling std of 1-bar log returns — known
    # at decision time t, so barrier width introduces no leakage.
    sigma = (
        pl.Series(ret1)
        .rolling_std(window_size=vol_lookback, min_samples=max(2, vol_lookback // 4))
        .to_numpy()
    )
    sigma = np.nan_to_num(sigma, nan=np.nanmedian(sigma[~np.isnan(sigma)]) if np.any(~np.isnan(sigma)) else 0.0)

    barrier_width = np.sqrt(max_hold) * sigma
    pt_level = close * (1 + pt_mult * barrier_width)
    sl_level = close * (1 - sl_mult * barrier_width)

    label = np.zeros(n, dtype=np.int8)
    barrier_type = np.array(["none"] * n, dtype=object)
    exit_bar_offset = np.full(n, -1, dtype=np.int32)

    usable = n - max_hold
    if usable > 0:
        high_windows = np.lib.stride_tricks.sliding_window_view(high[1:], max_hold)[:usable]
        low_windows = np.lib.stride_tricks.sliding_window_view(low[1:], max_hold)[:usable]

        pt_breach = high_windows >= pt_level[:usable, None]
        sl_breach = low_windows <= sl_level[:usable, None]

        pt_idx = _first_touch_index(pt_breach)
        sl_idx = _first_touch_index(sl_breach)

        pt_valid = pt_idx >= 0
        sl_valid = sl_idx >= 0
        # Both barriers touched inside the same bar: OHLC can't tell which
        # came first, so resolve by where that bar closed vs. entry instead
        # of always crediting the profit-take (an upward label bias in
        # exactly the high-volatility bars where it matters most).
        tie = pt_valid & sl_valid & (pt_idx == sl_idx)
        rows = np.arange(usable)
        tie_sign = np.sign(close[np.where(tie, rows + pt_idx + 1, rows)] - close[:usable])
        pt_first = pt_valid & (~sl_valid | (pt_idx < sl_idx) | (tie & (tie_sign > 0)))
        sl_first = sl_valid & (~pt_valid | (sl_idx < pt_idx) | (tie & (tie_sign < 0)))
        neither = ~pt_first & ~sl_first

        label[:usable] = np.where(pt_first, 1, np.where(sl_first, -1, 0))
        exit_bar_offset[:usable] = np.where(
            pt_first, pt_idx + 1, np.where(sl_first, sl_idx + 1, np.where(tie, pt_idx + 1, max_hold))
        )
        barrier_type[:usable] = np.where(pt_first, "profit_take", np.where(sl_first, "stop_loss", "time"))

        # Time-barrier outcomes get their direction from the sign of the
        # realized return at the max_hold mark instead of a flat 0.
        time_close = close[max_hold : max_hold + usable]
        time_ret = time_close - close[:usable]
        # A same-bar tie that closed exactly at entry stays a flat 0.
        label[:usable] = np.where(neither & ~tie, np.sign(time_ret).astype(np.int8), label[:usable])

    label[usable:] = 0
    barrier_type[usable:] = "insufficient_forward_data"
    exit_bar_offset[usable:] = -1

    out = df.with_columns(
        pl.Series("label_triple_barrier", label, dtype=pl.Int8),
        pl.Series("label_barrier_type", barrier_type),
        pl.Series("label_exit_bar_offset", exit_bar_offset, dtype=pl.Int32),
    )
    # Mark rows too close to the end of the available data as invalid for
    # training rather than silently keeping a degenerate label.
    valid_mask = pl.arange(0, n) < usable
    out = out.with_columns(valid_mask.alias("label_is_valid"))
    return out


def add_multi_horizon_return_labels(df: pl.DataFrame, close_col: str = "close") -> pl.DataFrame:
    n = df.height
    log_close = np.log(df[close_col].to_numpy())
    exprs = []
    for name, h in LABEL_HORIZONS_BARS.items():
        fwd = np.full(n, np.nan)
        if n > h:
            fwd[: n - h] = log_close[h:] - log_close[: n - h]
        exprs.append(pl.Series(f"label_return_{name}", fwd, dtype=pl.Float32))
    return df.with_columns(exprs)


def add_meta_label(df: pl.DataFrame) -> pl.DataFrame:
    """Secondary act/avoid label: 1 if the primary triple-barrier bet was
    decisive (a profit-take or stop-loss barrier was actually touched) and
    hence worth acting on; 0 if the bar drifted to the time barrier instead.

    This is a simplified meta-label appropriate at the feature-engineering
    stage, where no upstream primary-model side prediction exists yet — it
    flags "the market moved enough to resolve a bet" rather than filtering
    a specific model's predictions (that refinement belongs in the training
    pipeline, once real primary-model outputs are available).
    """
    return df.with_columns(
        (pl.col("label_barrier_type").is_in(["profit_take", "stop_loss"]) & pl.col("label_is_valid"))
        .cast(pl.Int8)
        .alias("label_meta_act")
    )


def add_all_labels(df: pl.DataFrame) -> pl.DataFrame:
    df = triple_barrier_labels(df)
    df = add_multi_horizon_return_labels(df)
    df = add_meta_label(df)
    return df


def relabel_contiguous(df: pl.DataFrame, bar_seconds: int = PRIMARY_INTERVAL_SECONDS) -> pl.DataFrame:
    """Recomputes every label over runs of contiguous bars spanning day files.

    build_dataset labels each day in isolation, so the last max_hold bars of
    every day are marked invalid and label_return_h4 (8640 bars == a full
    day) is NaN on every row. Once days are concatenated (common.data.
    load_days), labels can see across the midnight boundary — but only where
    the next day actually follows without a gap, so the frame is split at
    any ts step != bar_seconds and each run is labeled on its own.
    """
    if df.height == 0 or not all(c in df.columns for c in ("ts", "close", "high", "low")):
        return df
    step = pl.duration(seconds=bar_seconds) if df.schema["ts"] == pl.Datetime else bar_seconds * 1_000_000_000
    seg = (pl.col("ts").diff() != step).fill_null(True).cum_sum().alias("_seg")
    df = df.with_columns(seg)
    label_cols = [c for c in df.columns if c.startswith("label_")]
    parts = []
    for part in df.partition_by("_seg", maintain_order=True):
        parts.extend(_label_in_chunks(part.drop(label_cols)))
    return pl.concat(parts, how="vertical_relaxed").drop("_seg")


def _label_in_chunks(seg: pl.DataFrame, chunk: int = 4 * 8640) -> list[pl.DataFrame]:
    """add_all_labels over overlapping windows: each chunk gets
    TRIPLE_BARRIER_VOL_LOOKBACK bars of history (for sigma) and the longest
    forward horizon of future, so labels match a whole-segment pass while
    the (rows x max_hold) barrier matrices stay small."""
    ahead = max(TRIPLE_BARRIER_MAX_HOLD_BARS, *LABEL_HORIZONS_BARS.values())
    n, out = seg.height, []
    for start in range(0, n, chunk):
        lo = max(0, start - TRIPLE_BARRIER_VOL_LOOKBACK)
        stop = min(n, start + chunk)
        window = add_all_labels(seg.slice(lo, min(n, stop + ahead) - lo))
        out.append(window.slice(start - lo, stop - start))
    return out
