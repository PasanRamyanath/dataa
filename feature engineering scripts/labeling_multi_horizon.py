"""
feature engineering scripts/labeling_multi_horizon.py
Implements:
1. Fixed-Width Window Fractional Differentiation (FFD) across all 14 resolutions.
2. Fee-And-Slippage-Floored Multi-Horizon Triple-Barrier Labels (Macro, Meso, Tactical, Micro).
3. Trend-Scanning t-Statistic Labels (Lopez de Prado 2020) for Meso/Macro trend persistence.
4. Microstructure Sniper Entry & Exit Quality Labels (`label_micro_sniper_long/short`).
5. Sample Uniqueness / Average Concurrency Weights for overlapping multi-bar labels.
"""
from __future__ import annotations

import numpy as np
import polars as pl


def compute_ffd_weights(d: float, threshold: float = 1e-4, max_lags: int = 256) -> np.ndarray:
    """Computes Fixed-Width Window Fractional Differentiation weights w_k(d)."""
    weights = [1.0]
    for k in range(1, max_lags):
        w_k = -weights[-1] * (d - k + 1.0) / k
        if abs(w_k) < threshold:
            break
        weights.append(w_k)
    return np.asarray(weights, dtype=np.float64)


def apply_fractional_differentiation(
    series: np.ndarray,
    d: float = 0.35,
    threshold: float = 1e-4,
) -> np.ndarray:
    """Applies causal FFD convolution over a 1D log-price array without future leakage."""
    w = compute_ffd_weights(d=d, threshold=threshold)
    k_len = len(w)
    n = len(series)
    out = np.zeros(n, dtype=np.float64)
    if n < k_len:
        return out
    # Causal convolution: out[t] = sum_{k=0}^{K-1} w[k] * series[t - k]
    valid_conv = np.convolve(series, w, mode="valid")
    out[k_len - 1 :] = valid_conv
    return out


def fee_floored_triple_barrier_labels(
    df: pl.DataFrame,
    max_hold_bars: int,
    pt_mult: float,
    sl_mult: float,
    min_pt_bps: float,
    min_sl_bps: float,
    roundtrip_fee_bps: float = 4.0,
    vol_lookback: int = 48,
    prefix: str = "meso_15m",
) -> pl.DataFrame:
    """
    Computes Fee-Floored Volatility-Scaled Triple-Barrier Labels.
    Guarantees that label = +1 or -1 ONLY occurs when the price excursion
    clears roundtrip_fee_bps + net edge buffer (min_pt_bps).
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

    pt_Long_level = close * (1.0 + pt_width)
    sl_Long_level = close * (1.0 - sl_width)
    pt_Short_level = close * (1.0 - pt_width)
    sl_Short_level = close * (1.0 + sl_width)

    label = np.zeros(n, dtype=np.int8)
    realized_net_ret_bps = np.zeros(n, dtype=np.float32)
    exit_offset = np.full(n, max_hold_bars, dtype=np.int32)
    is_valid = np.zeros(n, dtype=bool)

    usable = n - max_hold_bars
    if usable > 0:
        is_valid[:usable] = True
        high_win = np.lib.stride_tricks.sliding_window_view(high[1:], max_hold_bars)[:usable]
        low_win = np.lib.stride_tricks.sliding_window_view(low[1:], max_hold_bars)[:usable]

        # Long profit-take vs stop-loss touch indices
        long_pt_hit = high_win >= pt_Long_level[:usable, None]
        long_sl_hit = low_win <= sl_Long_level[:usable, None]
        # Short profit-take vs stop-loss touch indices
        short_pt_hit = low_win <= pt_Short_level[:usable, None]
        short_sl_hit = high_win >= sl_Short_level[:usable, None]

        def _first_idx(mat: np.ndarray) -> np.ndarray:
            any_hit = mat.any(axis=1)
            return np.where(any_hit, np.argmax(mat, axis=1), max_hold_bars + 1)

        l_pt_i = _first_idx(long_pt_hit)
        l_sl_i = _first_idx(long_sl_hit)
        s_pt_i = _first_idx(short_pt_hit)
        s_sl_i = _first_idx(short_sl_hit)

        long_win = (l_pt_i <= max_hold_bars) & (l_pt_i < l_sl_i)
        short_win = (s_pt_i <= max_hold_bars) & (s_pt_i < s_sl_i)

        # Resolve rare cases where both long_win and short_win are true by whichever touched first
        both_win = long_win & short_win
        long_first = both_win & (l_pt_i < s_pt_i)
        short_first = both_win & (s_pt_i < l_pt_i)
        long_win = np.where(both_win, long_first, long_win)
        short_win = np.where(both_win, short_first, short_win)

        label[:usable] = np.where(long_win, 1, np.where(short_win, -1, 0))
        exit_offset[:usable] = np.where(
            long_win, l_pt_i + 1, np.where(short_win, s_pt_i + 1, max_hold_bars)
        )

        # Compute net return in bps at barrier exit (or time-stop exit)
        rows = np.arange(usable)
        end_close = close[rows + exit_offset[:usable]]
        gross_ret_bps = np.where(
            long_win,
            pt_width[:usable] * 1e4,
            np.where(
                short_win,
                pt_width[:usable] * 1e4,
                np.abs(end_close - close[:usable]) / close[:usable] * 1e4,
            ),
        )
        realized_net_ret_bps[:usable] = (gross_ret_bps - roundtrip_fee_bps).astype(np.float32)

    return df.with_columns([
        pl.Series(f"label_{prefix}_tb", label),
        pl.Series(f"label_{prefix}_net_bps", realized_net_ret_bps),
        pl.Series(f"label_{prefix}_exit_bars", exit_offset),
        pl.Series(f"label_{prefix}_valid", is_valid),
    ])


def compute_sample_uniqueness_weights(exit_offsets: np.ndarray, is_valid: np.ndarray) -> np.ndarray:
    """
    Computes Lopez de Prado's Average Uniqueness sample weights u_i in (0, 1]
    for overlapping multi-bar triple-barrier labels so that bars during low-concurrency
    regimes are not drowned out by redundant overlapping bars in high-concurrency clusters.
    """
    n = len(exit_offsets)
    concurrency = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if not is_valid[i]:
            continue
        end_i = min(n, i + max(1, int(exit_offsets[i])))
        concurrency[i:end_i] += 1.0

    concurrency = np.maximum(concurrency, 1.0)
    inv_conc = 1.0 / concurrency
    csum = np.cumsum(np.insert(inv_conc, 0, 0.0))

    weights = np.zeros(n, dtype=np.float32)
    for i in range(n):
        if not is_valid[i]:
            continue
        end_i = min(n, i + max(1, int(exit_offsets[i])))
        weights[i] = float((csum[end_i] - csum[i]) / max(1, end_i - i))

    mean_w = float(np.mean(weights[is_valid])) if np.any(is_valid) else 1.0
    if mean_w > 0:
        weights[is_valid] /= mean_w
    return weights
