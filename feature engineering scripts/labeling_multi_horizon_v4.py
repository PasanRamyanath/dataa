"""
feature engineering scripts/labeling_multi_horizon_v4.py
Remediates Audit Flaw #3: Computes exact signed net returns for both candidate
LONG and candidate SHORT trades under asymmetric Maker-TP (4.0 bps), Passive-Unwind
(5.0 bps), and Taker-SL (7.5 bps) execution friction.
"""
from __future__ import annotations

import numpy as np
import polars as pl


def fee_floored_triple_barrier_labels_v4(
    df: pl.DataFrame,
    max_hold_bars: int,
    pt_mult: float,
    sl_mult: float,
    min_pt_bps: float,
    min_sl_bps: float,
    maker_tp_fee_bps: float = 4.0,
    passive_unwind_fee_bps: float = 5.0,
    taker_sl_fee_bps: float = 7.5,
    vol_lookback: int = 48,
    prefix: str = "meso_15m",
) -> pl.DataFrame:
    """
    Computes Fee-Floored Triple-Barrier Labels AND exact signed net PnL (in bps)
    for both LONG and SHORT candidate actions at every bar `t`.
    """
    n = df.height
    close = df["close"].to_numpy().astype(np.float64)
    high = df["high"].to_numpy().astype(np.float64)
    low = df["low"].to_numpy().astype(np.float64)

    log_close = np.log(np.maximum(close, 1e-8))
    ret1 = np.diff(log_close, prepend=log_close[0])
    sigma = (
        pl.Series(ret1)
        .rolling_std(window_size=vol_lookback, min_samples=max(4, vol_lookback // 4))
        .to_numpy()
    )
    med_sigma = float(np.nanmedian(sigma[~np.isnan(sigma)])) if np.any(~np.isnan(sigma)) else 0.001
    sigma = np.nan_to_num(sigma, nan=med_sigma)

    raw_horizon_vol = np.sqrt(float(max_hold_bars)) * sigma
    pt_width = np.maximum(pt_mult * raw_horizon_vol, min_pt_bps * 1e-4)
    sl_width = np.maximum(sl_mult * raw_horizon_vol, min_sl_bps * 1e-4)

    pt_long_lvl = close * (1.0 + pt_width)
    sl_long_lvl = close * (1.0 - sl_width)
    pt_short_lvl = close * (1.0 - pt_width)
    sl_short_lvl = close * (1.0 + sl_width)

    label = np.zeros(n, dtype=np.int8)
    net_ret_long_bps = np.zeros(n, dtype=np.float32)
    net_ret_short_bps = np.zeros(n, dtype=np.float32)
    exit_offset = np.full(n, max_hold_bars, dtype=np.int32)
    is_valid = np.zeros(n, dtype=bool)

    usable = n - max_hold_bars
    if usable > 0:
        is_valid[:usable] = True
        high_win = np.lib.stride_tricks.sliding_window_view(high[1:], max_hold_bars)[:usable]
        low_win = np.lib.stride_tricks.sliding_window_view(low[1:], max_hold_bars)[:usable]

        long_pt_hit = high_win >= pt_long_lvl[:usable, None]
        long_sl_hit = low_win <= sl_long_lvl[:usable, None]
        short_pt_hit = low_win <= pt_short_lvl[:usable, None]
        short_sl_hit = high_win >= sl_short_lvl[:usable, None]

        def _first_idx(mat: np.ndarray) -> np.ndarray:
            any_hit = mat.any(axis=1)
            return np.where(any_hit, np.argmax(mat, axis=1), max_hold_bars + 1)

        l_pt_i = _first_idx(long_pt_hit)
        l_sl_i = _first_idx(long_sl_hit)
        s_pt_i = _first_idx(short_pt_hit)
        s_sl_i = _first_idx(short_sl_hit)

        # Evaluate exact outcome of a candidate LONG trade opened at bar t
        l_is_tp = (l_pt_i <= max_hold_bars) & (l_pt_i < l_sl_i)
        l_is_sl = (l_sl_i <= max_hold_bars) & (l_sl_i <= l_pt_i)
        # Evaluate exact outcome of a candidate SHORT trade opened at bar t
        s_is_tp = (s_pt_i <= max_hold_bars) & (s_pt_i < s_sl_i)
        s_is_sl = (s_sl_i <= max_hold_bars) & (s_sl_i <= s_pt_i)

        end_close = close[np.arange(usable) + max_hold_bars]
        hold_signed_bps = ((end_close - close[:usable]) / close[:usable]) * 1e4

        # Signed net return (in bps) if taking a LONG trade at bar t
        net_long = np.where(
            l_is_tp,
            pt_width[:usable] * 1e4 - maker_tp_fee_bps,
            np.where(
                l_is_sl,
                -(sl_width[:usable] * 1e4) - taker_sl_fee_bps,
                hold_signed_bps - passive_unwind_fee_bps,
            ),
        )
        # Signed net return (in bps) if taking a SHORT trade at bar t
        net_short = np.where(
            s_is_tp,
            pt_width[:usable] * 1e4 - maker_tp_fee_bps,
            np.where(
                s_is_sl,
                -(sl_width[:usable] * 1e4) - taker_sl_fee_bps,
                (-hold_signed_bps) - passive_unwind_fee_bps,
            ),
        )

        net_ret_long_bps[:usable] = net_long.astype(np.float32)
        net_ret_short_bps[:usable] = net_short.astype(np.float32)

        # Primary 3-class label: +1 if Long TP hit first (and net_long > 0), -1 if Short TP hit first
        both_tp = l_is_tp & s_is_tp
        l_win = np.where(both_tp, l_pt_i < s_pt_i, l_is_tp) & (net_long > 0.0)
        s_win = np.where(both_tp, s_pt_i < l_pt_i, s_is_tp) & (net_short > 0.0)

        label[:usable] = np.where(l_win, 1, np.where(s_win, -1, 0))
        exit_offset[:usable] = np.where(
            l_win, l_pt_i + 1, np.where(s_win, s_pt_i + 1, max_hold_bars)
        )

    return df.with_columns([
        pl.Series(f"label_{prefix}_tb", label),
        pl.Series(f"label_{prefix}_net_long_bps", net_ret_long_bps),
        pl.Series(f"label_{prefix}_net_short_bps", net_ret_short_bps),
        pl.Series(f"label_{prefix}_exit_bars", exit_offset),
        pl.Series(f"label_{prefix}_valid", is_valid),
    ])
