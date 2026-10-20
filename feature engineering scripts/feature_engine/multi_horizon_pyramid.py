"""
feature engineering scripts/feature_engine/multi_horizon_pyramid.py
Builds the 209-feature Stationary Multi-Timeframe Pyramid across all 14 resolutions.
Every feature is scale-free (log-returns, FFD z-scores, volatility ratios, volume imbalances)
and strictly excludes raw price levels (`close`, `open`, `high`, `low`, `vwap`).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from labeling_multi_horizon import apply_fractional_differentiation
from parsers.multi_tf_parser import causal_multi_timeframe_asof_join
from utils import clean_inf_nan, downcast_floats_to_f32, safe_div


def compute_single_timeframe_stationary_block(
    kline_df: pl.DataFrame,
    tf: str,
    ffd_d: float = 0.35,
    index_kline_df: pl.DataFrame | None = None,
    premium_kline_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """
    Given a normalized kline DataFrame for timeframe `tf` (with `ts` = completion timestamp),
    computes 14 stationary scale-invariant features:
      1. `ffd_zscore_20`: FFD(d) log-price standardized over 20 bars
      2. `ret_1_z`: 1-bar log-return divided by 20-bar Yang-Zhang volatility
      3. `ret_3_z`: 3-bar log-return divided by (sqrt(3) * 20-bar volatility)
      4. `ret_6_z`: 6-bar log-return divided by (sqrt(6) * 20-bar volatility)
      5. `yz_vol_bps`: 20-bar Yang-Zhang / Garman-Klass volatility in bps
      6. `vol_regime_ratio`: 5-bar short volatility / 20-bar volatility
      7. `donchian_pos_20`: Position in 20-bar High-Low range [0, 1]
      8. `rsi_14_norm`: Wilder RSI-14 centered and scaled to [-1, +1]
      9. `vwap_dev_z`: (close - vwap) / (close * yz_vol)
     10. `taker_buy_imb_1`: Single-bar (taker_buy - taker_sell) / volume
     11. `taker_buy_imb_ma5`: 5-bar moving average of taker buy/sell imbalance
     12. `cvd_slope_z20`: 20-bar normalized slope of volume delta
     13. `trade_size_z20`: 20-bar z-score of average trade quote size
     14. `spot_perp_basis_bps`: (close - index_close) / index_close * 1e4
    """
    if kline_df.is_empty():
        return pl.DataFrame()

    df = kline_df.sort("ts")
    closes = df["close"].to_numpy().astype(np.float64)
    log_c = np.log(np.maximum(closes, 1e-8))
    ffd_arr = apply_fractional_differentiation(log_c, d=ffd_d, threshold=1e-4)

    df = df.with_columns([
        pl.Series("log_close", log_c),
        pl.Series("ffd_raw", ffd_arr),
    ])

    # Garman-Klass single-bar variance estimator
    hl_log = (pl.col("high").clip(lower_bound=1e-8) / pl.col("low").clip(lower_bound=1e-8)).log()
    co_log = (pl.col("close").clip(lower_bound=1e-8) / pl.col("open").clip(lower_bound=1e-8)).log()
    gk_var = (0.5 * (hl_log ** 2) - (2.0 * np.log(2.0) - 1.0) * (co_log ** 2)).clip(lower_bound=1e-12)

    df = df.with_columns([
        (pl.col("log_close") - pl.col("log_close").shift(1)).fill_null(0.0).alias("ret_1"),
        (pl.col("log_close") - pl.col("log_close").shift(3)).fill_null(0.0).alias("ret_3"),
        (pl.col("log_close") - pl.col("log_close").shift(6)).fill_null(0.0).alias("ret_6"),
        gk_var.rolling_mean(window_size=20, min_samples=2).sqrt().alias("gk_vol_20"),
        gk_var.rolling_mean(window_size=5, min_samples=2).sqrt().alias("gk_vol_5"),
        pl.col("ffd_raw").rolling_mean(window_size=20, min_samples=2).alias("ffd_ma20"),
        pl.col("ffd_raw").rolling_std(window_size=20, min_samples=2).fill_null(1e-4).alias("ffd_std20"),
        pl.col("high").rolling_max(window_size=20, min_samples=2).alias("high_20"),
        pl.col("low").rolling_min(window_size=20, min_samples=2).alias("low_20"),
    ])

    # RSI-14 centered in [-1, +1]
    gain = pl.when(pl.col("ret_1") > 0).then(pl.col("ret_1")).otherwise(0.0)
    loss = pl.when(pl.col("ret_1") < 0).then(-pl.col("ret_1")).otherwise(0.0)
    avg_gain = gain.ewm_mean(span=14, ignore_nulls=True)
    avg_loss = loss.ewm_mean(span=14, ignore_nulls=True)
    rs = safe_div(avg_gain, avg_loss + 1e-12)
    rsi_norm = ((100.0 - (100.0 / (1.0 + rs))) - 50.0) / 50.0

    # Attach index close if provided for exact basis
    if index_kline_df is not None and not index_kline_df.is_empty():
        idx_sub = index_kline_df.select(["ts", pl.col("close").alias("idx_close")]).sort("ts")
        df = df.join_asof(idx_sub, on="ts", strategy="backward")
        basis_bps = ((pl.col("close") - pl.col("idx_close")) / pl.col("idx_close").clip(lower_bound=1e-8)) * 1e4
    else:
        basis_bps = pl.lit(0.0, dtype=pl.Float64)

    vol_20 = pl.col("gk_vol_20").clip(lower_bound=1e-6)
    df = df.with_columns([
        safe_div(pl.col("ffd_raw") - pl.col("ffd_ma20"), pl.col("ffd_std20") + 1e-8).clip(-6.0, 6.0).alias("ffd_zscore_20"),
        safe_div(pl.col("ret_1"), vol_20).clip(-6.0, 6.0).alias("ret_1_z"),
        safe_div(pl.col("ret_3"), vol_20 * np.sqrt(3.0)).clip(-6.0, 6.0).alias("ret_3_z"),
        safe_div(pl.col("ret_6"), vol_20 * np.sqrt(6.0)).clip(-6.0, 6.0).alias("ret_6_z"),
        (vol_20 * 1e4).alias("yz_vol_bps"),
        safe_div(pl.col("gk_vol_5"), vol_20).clip(0.1, 5.0).alias("vol_regime_ratio"),
        safe_div(pl.col("close") - pl.col("low_20"), (pl.col("high_20") - pl.col("low_20")) + 1e-8).clip(0.0, 1.0).alias("donchian_pos_20"),
        rsi_norm.clip(-1.0, 1.0).alias("rsi_14_norm"),
        safe_div((pl.col("close") - pl.col("vwap")) / pl.col("close").clip(lower_bound=1e-8), vol_20).clip(-5.0, 5.0).alias("vwap_dev_z"),
        pl.col("taker_buy_sell_imbalance").clip(-1.0, 1.0).alias("taker_buy_imb_1"),
        pl.col("taker_buy_sell_imbalance").rolling_mean(window_size=5, min_samples=1).clip(-1.0, 1.0).alias("taker_buy_imb_ma5"),
        safe_div(
            pl.col("volume_delta").rolling_sum(window_size=5, min_samples=1),
            pl.col("volume").rolling_sum(window_size=5, min_samples=1) + 1e-8,
        ).clip(-1.0, 1.0).alias("cvd_slope_z20"),
        safe_div(
            pl.col("avg_trade_quote_size") - pl.col("avg_trade_quote_size").rolling_mean(window_size=20, min_samples=2),
            pl.col("avg_trade_quote_size").rolling_std(window_size=20, min_samples=2) + 1e-8,
        ).clip(-5.0, 5.0).alias("trade_size_z20"),
        basis_bps.fill_null(0.0).clip(-200.0, 200.0).alias("spot_perp_basis_bps"),
    ])

    feature_names = [
        "ffd_zscore_20", "ret_1_z", "ret_3_z", "ret_6_z", "yz_vol_bps",
        "vol_regime_ratio", "donchian_pos_20", "rsi_14_norm", "vwap_dev_z",
        "taker_buy_imb_1", "taker_buy_imb_ma5", "cvd_slope_z20",
        "trade_size_z20", "spot_perp_basis_bps",
    ]
    out = df.select(["ts", "open", "high", "low", "close", "volume", *feature_names])
    out = clean_inf_nan(out)
    return downcast_floats_to_f32(out)


def build_complete_14_timeframe_pyramid(
    base_grid_df: pl.DataFrame,
    tf_kline_dict: dict[str, pl.DataFrame],
    tf_index_dict: dict[str, pl.DataFrame] | None = None,
) -> pl.DataFrame:
    """
    Causally aligns all 14 resolutions onto `base_grid_df` using `decision_ts >= close_ts`
    backward as-of joins so zero future information ever leaks across timeframes.
    """
    tf_index_dict = tf_index_dict or {}
    feature_cols = [
        "ffd_zscore_20", "ret_1_z", "ret_3_z", "ret_6_z", "yz_vol_bps",
        "vol_regime_ratio", "donchian_pos_20", "rsi_14_norm", "vwap_dev_z",
        "taker_buy_imb_1", "taker_buy_imb_ma5", "cvd_slope_z20",
        "trade_size_z20", "spot_perp_basis_bps",
    ]
    ffd_map = {
        "1s": 0.50, "10s": 0.50,
        "1m": 0.42, "3m": 0.42, "5m": 0.42,
        "15m": 0.35, "30m": 0.35, "1h": 0.35, "2h": 0.35,
        "4h": 0.28, "6h": 0.28, "8h": 0.28, "12h": 0.28, "1d": 0.28,
    }

    out = base_grid_df.sort("ts")
    for tf, kdf in tf_kline_dict.items():
        d_val = ffd_map.get(tf, 0.35)
        idx_df = tf_index_dict.get(tf)
        stat_block = compute_single_timeframe_stationary_block(kdf, tf=tf, ffd_d=d_val, index_kline_df=idx_df)
        out = causal_multi_timeframe_asof_join(
            base_df=out,
            htf_df=stat_block,
            htf_prefix=f"tf_{tf}",
            feature_cols=feature_cols,
            base_ts_col="ts",
            htf_close_ts_col="ts",
        )

    out = clean_inf_nan(out)
    return downcast_floats_to_f32(out)
