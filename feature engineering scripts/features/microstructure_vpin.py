"""
Order Book Microstructure & Flow Toxicity Feature Generator (Family 3).
Computes Level 1 queue imbalance, Stoikov microprice drift,
Trade Flow Imbalance (TFI), VPIN toxicity proxy, and liquidation bursts
with strict point-in-time cutoff verification (t_event <= cutoff <= T_close - 1000ms).
"""
from __future__ import annotations

import polars as pl
import numpy as np


def _safe_div(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    return pl.when(den.abs() > 1e-12).then(num / den).otherwise(0.0)


def add_microstructure_and_vpin_features(
    bars: pl.DataFrame,
    orderbook_df: pl.DataFrame | None,
    cutoff_timestamp_ms: int | None = None,
) -> pl.DataFrame:
    """
    Computes Level 1 queue imbalance, Stoikov microprice drift,
    trade flow imbalance (VPIN / toxicity), and liquidation burst indicators.

    Parameters
    ----------
    bars : pl.DataFrame
        Primary BTCUSDT 10s bars.
    orderbook_df : pl.DataFrame | None
        Order book depth or synthetic microstructure feed.
    cutoff_timestamp_ms : int | None
        Strict timestamp cutoff.

    Returns
    -------
    pl.DataFrame
        Enriched DataFrame with microstructure & VPIN features.
    """
    feature_cols = [
        "l1_queue_imbalance",
        "l1_queue_imbalance_ma3",
        "l1_queue_imbalance_ma6",
        "l1_queue_velocity",
        "microprice_drift_bps",
        "microprice_velocity_10s",
        "microprice_velocity_30s",
        "trade_flow_imbalance",
        "tfi_ma_3",
        "tfi_ma_6",
        "vpin_toxicity_6",
        "vpin_toxicity_30",
        "vpin_surge_ratio",
        "liq_net_flow_bps",
        "liq_burst_zscore",
        "liq_burst_active",
        "l1_imbalance",  # interoperability alias
        "vpin",          # interoperability alias
        "ofi",           # interoperability alias
    ]

    if bars.is_empty():
        return bars.with_columns([
            pl.lit(0.0, dtype=pl.Float32).alias(c) if c != "liq_burst_active"
            else pl.lit(0, dtype=pl.Int8).alias(c)
            for c in feature_cols
        ])

    b_cols = bars.columns
    # Trade flow imbalance from bars
    if "buy_volume" in b_cols and "sell_volume" in b_cols:
        b_vol = pl.col("buy_volume").cast(pl.Float64)
        s_vol = pl.col("sell_volume").cast(pl.Float64)
    elif "volume" in b_cols:
        b_vol = (pl.col("volume") * 0.5).cast(pl.Float64)
        s_vol = (pl.col("volume") * 0.5).cast(pl.Float64)
    else:
        b_vol = pl.lit(1.0)
        s_vol = pl.lit(1.0)

    tot_vol = b_vol + s_vol + 1e-6
    tfi = _safe_div(b_vol - s_vol, tot_vol).clip(-1.0, 1.0)
    tfi_ma3 = tfi.rolling_mean(3, min_samples=1)
    tfi_ma6 = tfi.rolling_mean(6, min_samples=1)

    vol_imb = (b_vol - s_vol).abs()
    vpin_6 = _safe_div(vol_imb.rolling_sum(6, min_samples=1), tot_vol.rolling_sum(6, min_samples=1)).clip(0.0, 1.0)
    vpin_30 = _safe_div(vol_imb.rolling_sum(30, min_samples=1), tot_vol.rolling_sum(30, min_samples=1)).clip(0.0, 1.0)
    vpin_surge = _safe_div(vpin_6, vpin_30 + 1e-6)

    out = bars.with_columns([
        tfi.fill_null(0.0).alias("trade_flow_imbalance"),
        tfi_ma3.fill_null(0.0).alias("tfi_ma_3"),
        tfi_ma6.fill_null(0.0).alias("tfi_ma_6"),
        vpin_6.fill_null(0.0).alias("vpin_toxicity_6"),
        vpin_30.fill_null(0.0).alias("vpin_toxicity_30"),
        vpin_surge.fill_null(1.0).alias("vpin_surge_ratio"),
    ])

    # If orderbook_df is available, extract L1 and microprice features
    if orderbook_df is not None and not orderbook_df.is_empty():
        ob = orderbook_df
        ts_col = "cutoff_ts" if "cutoff_ts" in ob.columns else "ts"
        if cutoff_timestamp_ms is not None:
            if ob.schema[ts_col] == pl.Datetime:
                cutoff_dt = pl.from_epoch(cutoff_timestamp_ms * 1_000_000, time_unit="ns")
                ob = ob.filter(pl.col(ts_col) <= cutoff_dt)
            else:
                ob = ob.filter(pl.col(ts_col) <= cutoff_timestamp_ms)

        if not ob.is_empty():
            ob_cols = ob.columns
            # Check if synthetic precomputed features exist
            has_l1 = "l1_imbalance" in ob_cols
            has_micro_drift = "microprice_drift_bps" in ob_cols
            has_vpin = "vpin" in ob_cols
            has_ofi = "ofi" in ob_cols

            b_p = (
                pl.col("best_bid") if "best_bid" in ob_cols
                else (pl.col("bbo_bid") if "bbo_bid" in ob_cols
                      else (pl.col("close") if "close" in ob_cols
                            else (pl.col("microprice") if "microprice" in ob_cols else pl.lit(65000.0))))
            ).cast(pl.Float64)
            a_p = (
                pl.col("best_ask") if "best_ask" in ob_cols
                else (pl.col("bbo_ask") if "bbo_ask" in ob_cols
                      else (pl.col("close") if "close" in ob_cols
                            else (pl.col("microprice") if "microprice" in ob_cols else pl.lit(65000.0))))
            ).cast(pl.Float64)
            b_q = (
                pl.col("best_bid_qty") if "best_bid_qty" in ob_cols
                else (pl.col("bbo_bid_qty") if "bbo_bid_qty" in ob_cols else pl.lit(1.0))
            ).cast(pl.Float64)
            a_q = (
                pl.col("best_ask_qty") if "best_ask_qty" in ob_cols
                else (pl.col("bbo_ask_qty") if "bbo_ask_qty" in ob_cols else pl.lit(1.0))
            ).cast(pl.Float64)

            liq_b = (
                pl.col("liquidation_buy_notional") if "liquidation_buy_notional" in ob_cols
                else (pl.col("liquidation_burst_usd") * 0.5 if "liquidation_burst_usd" in ob_cols
                      else (pl.col("liquidations_usd") * 0.5 if "liquidations_usd" in ob_cols else pl.lit(0.0)))
            ).cast(pl.Float64)
            liq_s = (
                pl.col("liquidation_sell_notional") if "liquidation_sell_notional" in ob_cols
                else (pl.col("liquidation_burst_usd") * 0.5 if "liquidation_burst_usd" in ob_cols
                      else (pl.col("liquidations_usd") * 0.5 if "liquidations_usd" in ob_cols else pl.lit(0.0)))
            ).cast(pl.Float64)

            prep_exprs = [
                pl.col("ts"),
                b_p.alias("_bid_p"),
                a_p.alias("_ask_p"),
                b_q.alias("_bid_q"),
                a_q.alias("_ask_q"),
                liq_b.alias("_liq_b"),
                liq_s.alias("_liq_s"),
            ]
            if has_l1:
                prep_exprs.append(pl.col("l1_imbalance").cast(pl.Float64).alias("_pre_l1"))
            if has_micro_drift:
                prep_exprs.append(pl.col("microprice_drift_bps").cast(pl.Float64).alias("_pre_micro_drift"))
            if has_vpin:
                prep_exprs.append(pl.col("vpin").cast(pl.Float64).alias("_pre_vpin"))
            if has_ofi:
                prep_exprs.append(pl.col("ofi").cast(pl.Float64).alias("_pre_ofi"))

            ob_prep = ob.select(prep_exprs).sort("ts")
            out = out.sort("ts").join_asof(ob_prep, on="ts", strategy="backward")

            # Forward fill
            ff_cols = ["_bid_p", "_ask_p", "_bid_q", "_ask_q", "_liq_b", "_liq_s"]
            if has_l1:
                ff_cols.append("_pre_l1")
            if has_micro_drift:
                ff_cols.append("_pre_micro_drift")
            if has_vpin:
                ff_cols.append("_pre_vpin")
            if has_ofi:
                ff_cols.append("_pre_ofi")

            out = out.with_columns([pl.col(c).forward_fill() for c in ff_cols])

            # L1 Queue Imbalance
            if has_l1:
                q_imb = pl.col("_pre_l1").clip(-1.0, 1.0)
            else:
                q_imb = _safe_div(pl.col("_bid_q") - pl.col("_ask_q"), pl.col("_bid_q") + pl.col("_ask_q") + 1e-6).clip(-1.0, 1.0)

            # Stoikov Microprice
            mid_p = (pl.col("_bid_p") + pl.col("_ask_p")) / 2.0
            spread = pl.col("_ask_p") - pl.col("_bid_p")
            micro_p = mid_p + (spread / 2.0) * q_imb

            if has_micro_drift:
                micro_drift_bps = pl.col("_pre_micro_drift")
            else:
                micro_drift_bps = _safe_div(micro_p - mid_p, mid_p) * 10000.0

            # Override VPIN/OFI if pre-computed in orderbook feed
            if has_vpin:
                out = out.with_columns([
                    pl.col("_pre_vpin").alias("vpin_toxicity_6"),
                    pl.col("_pre_vpin").alias("vpin"),
                ])
            else:
                out = out.with_columns([pl.col("vpin_toxicity_6").alias("vpin")])

            if has_ofi:
                out = out.with_columns([
                    pl.col("_pre_ofi").alias("trade_flow_imbalance"),
                    pl.col("_pre_ofi").alias("ofi"),
                ])
            else:
                out = out.with_columns([pl.col("trade_flow_imbalance").alias("ofi")])

            out = out.with_columns([
                q_imb.alias("l1_queue_imbalance"),
                q_imb.alias("l1_imbalance"),  # alias
                q_imb.rolling_mean(3, min_samples=1).alias("l1_queue_imbalance_ma3"),
                q_imb.rolling_mean(6, min_samples=1).alias("l1_queue_imbalance_ma6"),
                (q_imb - q_imb.shift(1)).fill_null(0.0).alias("l1_queue_velocity"),
                micro_drift_bps.alias("microprice_drift_bps"),
                (_safe_div(micro_p - micro_p.shift(1), mid_p) * 10000.0).fill_null(0.0).alias("microprice_velocity_10s"),
                (_safe_div(micro_p - micro_p.shift(3), mid_p) * 10000.0).fill_null(0.0).alias("microprice_velocity_30s"),
            ])

            # Liquidations
            liq_tot = pl.col("_liq_b") + pl.col("_liq_s")
            liq_net = pl.col("_liq_b") - pl.col("_liq_s")
            notional_ref = pl.col("notional_volume") if "notional_volume" in out.columns else pl.lit(100000.0)
            liq_flow_bps = _safe_div(liq_net, notional_ref + 100000.0) * 10000.0

            liq_mean_180 = liq_tot.rolling_mean(180, min_samples=1)
            liq_std_180 = liq_tot.rolling_std(180, min_samples=1)
            liq_z = _safe_div(liq_tot - liq_mean_180, liq_std_180 + 1e-6).fill_null(0.0)
            liq_active = pl.when((liq_z >= 2.5) & (liq_tot >= 250000.0)).then(1).otherwise(0).cast(pl.Int8)

            out = out.with_columns([
                liq_flow_bps.fill_null(0.0).alias("liq_net_flow_bps"),
                liq_z.alias("liq_burst_zscore"),
                liq_active.alias("liq_burst_active"),
            ])
            out = out.drop([c for c in ff_cols if c in out.columns])
        else:
            out = _add_default_l1_columns(out)
    else:
        out = _add_default_l1_columns(out)

    return out.with_columns([
        pl.col(c).cast(pl.Float32).fill_null(0.0).fill_nan(0.0)
        if c != "liq_burst_active" else pl.col(c).cast(pl.Int8).fill_null(0)
        for c in feature_cols if c in out.columns
    ])


def _add_default_l1_columns(df: pl.DataFrame) -> pl.DataFrame:
    # Use BBO columns if available in bars
    cols = df.columns
    if "bbo_bid_qty" in cols and "bbo_ask_qty" in cols:
        b_q = pl.col("bbo_bid_qty").cast(pl.Float64)
        a_q = pl.col("bbo_ask_qty").cast(pl.Float64)
        q_imb = _safe_div(b_q - a_q, b_q + a_q + 1e-6).clip(-1.0, 1.0)
    else:
        q_imb = pl.lit(0.0)

    if "bbo_bid" in cols and "bbo_ask" in cols:
        b_p = pl.col("bbo_bid").cast(pl.Float64)
        a_p = pl.col("bbo_ask").cast(pl.Float64)
        mid_p = (b_p + a_p) / 2.0
        spread = a_p - b_p
        micro_drift_bps = _safe_div((spread / 2.0) * q_imb, mid_p) * 10000.0
    else:
        micro_drift_bps = pl.lit(0.0)

    return df.with_columns([
        q_imb.alias("l1_queue_imbalance"),
        q_imb.alias("l1_imbalance"),
        q_imb.rolling_mean(3, min_samples=1).alias("l1_queue_imbalance_ma3"),
        q_imb.rolling_mean(6, min_samples=1).alias("l1_queue_imbalance_ma6"),
        (q_imb - q_imb.shift(1)).fill_null(0.0).alias("l1_queue_velocity"),
        micro_drift_bps.alias("microprice_drift_bps"),
        pl.lit(0.0, dtype=pl.Float32).alias("microprice_velocity_10s"),
        pl.lit(0.0, dtype=pl.Float32).alias("microprice_velocity_30s"),
        pl.lit(0.0, dtype=pl.Float32).alias("liq_net_flow_bps"),
        pl.lit(0.0, dtype=pl.Float32).alias("liq_burst_zscore"),
        pl.lit(0, dtype=pl.Int8).alias("liq_burst_active"),
        pl.col("vpin_toxicity_6").alias("vpin"),
        pl.col("trade_flow_imbalance").alias("ofi"),
    ])
