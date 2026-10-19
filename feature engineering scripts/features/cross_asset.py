"""
Cross-Asset Lead-Lag Dynamics Feature Generator (Family 1).
Computes ETHUSDT vs BTCUSDT lead-lag order flow, returns, and momentum
over 10s, 30s, and 60s windows with strict point-in-time timestamp integrity
(t_event <= cutoff_timestamp_ms <= T_close - 1000ms).
"""
from __future__ import annotations

import polars as pl
import numpy as np

LEAD_LAGS = [1, 2, 3, 6]  # 10s, 20s, 30s, 60s at 10s bar resolution
SPREAD_WINDOWS = [1, 3, 6]  # 10s, 30s, 60s windows


def _safe_div(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    return pl.when(den.abs() > 1e-12).then(num / den).otherwise(0.0)


def add_cross_asset_lead_lag_features(
    btc_bars: pl.DataFrame,
    eth_bars: pl.DataFrame | None,
    cutoff_timestamp_ms: int | None = None,
) -> pl.DataFrame:
    """
    Computes cross-asset lead-lag features between ETHUSDT and BTCUSDT.

    Parameters
    ----------
    btc_bars : pl.DataFrame
        Primary BTCUSDT 10s bar DataFrame.
    eth_bars : pl.DataFrame | None
        Auxiliary ETHUSDT 10s bar DataFrame.
    cutoff_timestamp_ms : int | None
        Strict timestamp cutoff. Any data timestamped > cutoff_timestamp_ms is dropped.

    Returns
    -------
    pl.DataFrame
        Enriched DataFrame with cross-asset features.
    """
    feature_cols = (
        [f"eth_lead_return_lag_{k}" for k in LEAD_LAGS]
        + [f"cross_return_spread_{w}0s" for w in SPREAD_WINDOWS]
        + ["cross_return_residual_10s"]
        + ["eth_ofi"]
        + [f"eth_ofi_lag_{k}" for k in [1, 2, 3]]
        + [f"cross_ofi_divergence_{w}0s" for w in SPREAD_WINDOWS]
        + [f"cross_mom_diff_{w}0s" for w in SPREAD_WINDOWS]
        + ["cross_mom_diff_zscore_30s"]
        + ["eth_lead_ret_bps", "eth_imbalance"]  # interoperability aliases
    )

    if btc_bars.is_empty():
        return btc_bars.with_columns([pl.lit(0.0, dtype=pl.Float32).alias(c) for c in feature_cols])

    if eth_bars is None or eth_bars.is_empty():
        return btc_bars.with_columns([pl.lit(0.0, dtype=pl.Float32).alias(c) for c in feature_cols])

    # Enforce strict timestamp cutoff on auxiliary ETH stream
    eth = eth_bars
    ts_col = "cutoff_ts" if "cutoff_ts" in eth.columns else "ts"
    if cutoff_timestamp_ms is not None:
        if eth.schema[ts_col] == pl.Datetime:
            cutoff_dt = pl.from_epoch(cutoff_timestamp_ms * 1_000_000, time_unit="ns")
            eth = eth.filter(pl.col(ts_col) <= cutoff_dt)
        else:
            eth = eth.filter(pl.col(ts_col) <= cutoff_timestamp_ms)

    if eth.is_empty():
        return btc_bars.with_columns([pl.lit(0.0, dtype=pl.Float32).alias(c) for c in feature_cols])

    # Standardize column mappings for ETH
    eth_cols = eth.columns
    close_expr = (
        pl.col("eth_close") if "eth_close" in eth_cols
        else (pl.col("close") if "close" in eth_cols else pl.col("bbo_mid"))
    )
    buy_v_expr = (
        pl.col("buy_volume") if "buy_volume" in eth_cols
        else (pl.col("eth_volume") * 0.5 if "eth_volume" in eth_cols else pl.lit(1.0))
    )
    sell_v_expr = (
        pl.col("sell_volume") if "sell_volume" in eth_cols
        else (pl.col("eth_volume") * 0.5 if "eth_volume" in eth_cols else pl.lit(1.0))
    )

    # Compute ETH OFI
    if "eth_imbalance" in eth_cols:
        ofi_expr = pl.col("eth_imbalance")
    else:
        ofi_expr = _safe_div(buy_v_expr - sell_v_expr, buy_v_expr + sell_v_expr + 1e-6)

    eth_prep = eth.select([
        pl.col("ts"),
        close_expr.cast(pl.Float64).alias("_eth_close"),
        ofi_expr.cast(pl.Float64).alias("_eth_ofi"),
    ]).sort("ts")

    joined = btc_bars.sort("ts").join_asof(eth_prep, on="ts", strategy="backward")
    joined = joined.with_columns([
        pl.col("_eth_close").forward_fill(),
        pl.col("_eth_ofi").forward_fill().fill_null(0.0),
    ])

    btc_cols = btc_bars.columns
    btc_close = (
        pl.col("close") if "close" in btc_cols
        else (pl.col("bbo_mid") if "bbo_mid" in btc_cols else pl.lit(1.0))
    ).cast(pl.Float64)

    # Log returns
    eth_ret = pl.col("_eth_close").log().diff().fill_null(0.0)
    btc_ret = btc_close.log().diff().fill_null(0.0)

    exprs = []

    # 1. Lagged ETH Returns (in bps)
    for k in LEAD_LAGS:
        exprs.append((eth_ret.shift(k).fill_null(0.0) * 10000.0).alias(f"eth_lead_return_lag_{k}"))

    # 2. Return Spreads (in bps)
    for w in SPREAD_WINDOWS:
        w_eth = (pl.col("_eth_close").log() - pl.col("_eth_close").log().shift(w)).fill_null(0.0)
        w_btc = (btc_close.log() - btc_close.log().shift(w)).fill_null(0.0)
        exprs.append(((w_eth - w_btc) * 10000.0).alias(f"cross_return_spread_{w}0s"))

    # 3. Lagged ETH OFI
    exprs.append(pl.col("_eth_ofi").alias("eth_ofi"))
    for k in [1, 2, 3]:
        exprs.append(pl.col("_eth_ofi").shift(k).fill_null(0.0).alias(f"eth_ofi_lag_{k}"))

    # 4. BTC OFI
    if "buy_volume" in btc_cols and "sell_volume" in btc_cols:
        btc_ofi = _safe_div(
            pl.col("buy_volume") - pl.col("sell_volume"),
            pl.col("buy_volume") + pl.col("sell_volume") + 1e-6,
        ).fill_null(0.0)
    elif "bbo_bid_qty" in btc_cols and "bbo_ask_qty" in btc_cols:
        btc_ofi = _safe_div(
            pl.col("bbo_bid_qty") - pl.col("bbo_ask_qty"),
            pl.col("bbo_bid_qty") + pl.col("bbo_ask_qty") + 1e-6,
        ).fill_null(0.0)
    else:
        btc_ofi = pl.lit(0.0)

    # 5. OFI Divergence
    for w in SPREAD_WINDOWS:
        roll_eth_ofi = pl.col("_eth_ofi").rolling_mean(window_size=w, min_samples=1).fill_null(0.0)
        roll_btc_ofi = btc_ofi.rolling_mean(window_size=w, min_samples=1).fill_null(0.0)
        exprs.append((roll_eth_ofi - roll_btc_ofi).alias(f"cross_ofi_divergence_{w}0s"))

    # 6. Momentum Differentials (bps)
    for w in SPREAD_WINDOWS:
        eth_mom = _safe_div(pl.col("_eth_close") - pl.col("_eth_close").shift(w), pl.col("_eth_close").shift(w)) * 10000.0
        btc_mom = _safe_div(btc_close - btc_close.shift(w), btc_close.shift(w)) * 10000.0
        exprs.append((eth_mom.fill_null(0.0) - btc_mom.fill_null(0.0)).alias(f"cross_mom_diff_{w}0s"))

    out = joined.with_columns(exprs)

    # 7. Beta-adjusted residual return & standardized momentum z-score
    roll_cov = (
        (btc_ret - btc_ret.rolling_mean(180, min_samples=1))
        * (eth_ret - eth_ret.rolling_mean(180, min_samples=1))
    ).rolling_mean(180, min_samples=1)
    roll_var = eth_ret.rolling_std(180, min_samples=1).pow(2)
    beta = _safe_div(roll_cov, roll_var).fill_null(1.0)
    residual = (btc_ret - beta * eth_ret) * 10000.0

    mom_diff_30s = pl.col("cross_mom_diff_30s")
    mom_zscore = _safe_div(
        mom_diff_30s - mom_diff_30s.rolling_mean(360, min_samples=1),
        mom_diff_30s.rolling_std(360, min_samples=1) + 1e-6,
    )

    out = out.with_columns([
        residual.fill_null(0.0).alias("cross_return_residual_10s"),
        mom_zscore.fill_null(0.0).alias("cross_mom_diff_zscore_30s"),
        pl.col("eth_lead_return_lag_1").alias("eth_lead_ret_bps"),
        pl.col("eth_ofi").alias("eth_imbalance"),
    ])

    out = out.drop(["_eth_close", "_eth_ofi"])
    return out.with_columns([
        pl.col(c).cast(pl.Float32).fill_null(0.0).fill_nan(0.0)
        for c in feature_cols if c in out.columns
    ])
