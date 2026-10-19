"""
Spot-Perpetual Basis Divergence & Funding Rate Pressure Generator (Family 2).
Computes basis spread (bps), rolling z-scores, velocity, acceleration,
and forward funding carry pressure with zero lookahead leakage
(t_event <= cutoff_timestamp_ms <= T_close - 1000ms).
"""
from __future__ import annotations

import polars as pl
import numpy as np

BASIS_Z_WINDOWS = [180, 720, 2880]  # 30m, 2h, 8h @ 10s bars
VELOCITY_LAGS = [1, 3, 6]  # 10s, 30s, 60s


def _safe_div(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    return pl.when(den.abs() > 1e-12).then(num / den).otherwise(0.0)


def add_basis_and_funding_features(
    bars: pl.DataFrame,
    spot_perp_df: pl.DataFrame | None,
    cutoff_timestamp_ms: int | None = None,
) -> pl.DataFrame:
    """
    Computes spot-perp basis spread, rolling z-scores, velocity, acceleration,
    and funding rate pressure indicators.

    Parameters
    ----------
    bars : pl.DataFrame
        BTCUSDT perpetual bars.
    spot_perp_df : pl.DataFrame | None
        Spot-Perp feed containing mid-prices, basis, or funding rate.
    cutoff_timestamp_ms : int | None
        Strict timestamp cutoff.

    Returns
    -------
    pl.DataFrame
        Enriched DataFrame with basis & funding features.
    """
    feature_cols = (
        ["basis_spread_bps"]
        + [f"basis_zscore_{w}" for w in BASIS_Z_WINDOWS]
        + [f"basis_velocity_{k}0s" for k in VELOCITY_LAGS]
        + ["basis_acceleration_30s"]
        + ["funding_rate_raw", "funding_rate_ma_8h", "funding_pressure_bps", "funding_basis_divergence"]
        + ["funding_rate_pressure"]  # Interoperability alias
    )

    if bars.is_empty():
        return bars.with_columns([pl.lit(0.0, dtype=pl.Float32).alias(c) for c in feature_cols])

    if spot_perp_df is None or spot_perp_df.is_empty():
        return bars.with_columns([pl.lit(0.0, dtype=pl.Float32).alias(c) for c in feature_cols])

    sp = spot_perp_df
    ts_col = "cutoff_ts" if "cutoff_ts" in sp.columns else "ts"
    if cutoff_timestamp_ms is not None:
        if sp.schema[ts_col] == pl.Datetime:
            cutoff_dt = pl.from_epoch(cutoff_timestamp_ms * 1_000_000, time_unit="ns")
            sp = sp.filter(pl.col(ts_col) <= cutoff_dt)
        else:
            sp = sp.filter(pl.col(ts_col) <= cutoff_timestamp_ms)

    if sp.is_empty():
        return bars.with_columns([pl.lit(0.0, dtype=pl.Float32).alias(c) for c in feature_cols])

    cols = sp.columns
    p_mid = (
        pl.col("perp_mid") if "perp_mid" in cols
        else (pl.col("perp_close") if "perp_close" in cols
              else (pl.col("close") if "close" in cols else pl.col("bbo_mid")))
    ).cast(pl.Float64)

    s_mid = (
        pl.col("spot_mid") if "spot_mid" in cols
        else (pl.col("spot_close") if "spot_close" in cols else p_mid)
    ).cast(pl.Float64)

    f_rate = (pl.col("funding_rate") if "funding_rate" in cols else pl.lit(0.0)).cast(pl.Float64)
    nft = (pl.col("next_funding_time_ms") if "next_funding_time_ms" in cols else pl.lit(0)).cast(pl.Int64)

    # Check if pre-computed basis_spread_bps is already available in spot_perp_df
    has_precomputed_basis = "basis_spread_bps" in cols

    sp_prep = sp.select([
        pl.col("ts"),
        p_mid.alias("_p_mid"),
        s_mid.alias("_s_mid"),
        f_rate.alias("_funding_rate"),
        nft.alias("_next_funding_time_ms"),
        *( [pl.col("basis_spread_bps").cast(pl.Float64).alias("_pre_basis")] if has_precomputed_basis else [] )
    ]).sort("ts")

    out = bars.sort("ts").join_asof(sp_prep, on="ts", strategy="backward")
    out = out.with_columns([
        pl.col("_p_mid").forward_fill(),
        pl.col("_s_mid").forward_fill(),
        pl.col("_funding_rate").forward_fill().fill_null(0.0),
        pl.col("_next_funding_time_ms").forward_fill().fill_null(0),
        *( [pl.col("_pre_basis").forward_fill().fill_null(0.0)] if has_precomputed_basis else [] )
    ])

    # 1. Basis spread in bps
    if has_precomputed_basis:
        basis_bps = pl.col("_pre_basis")
    else:
        basis_bps = _safe_div(pl.col("_p_mid") - pl.col("_s_mid"), pl.col("_s_mid")) * 10000.0

    out = out.with_columns(basis_bps.fill_null(0.0).alias("basis_spread_bps"))

    # 2. Rolling Z-scores
    z_exprs = []
    for w in BASIS_Z_WINDOWS:
        roll_m = pl.col("basis_spread_bps").rolling_mean(w, min_samples=1)
        roll_s = pl.col("basis_spread_bps").rolling_std(w, min_samples=1)
        z_exprs.append(_safe_div(pl.col("basis_spread_bps") - roll_m, roll_s + 1e-6).fill_null(0.0).alias(f"basis_zscore_{w}"))
    out = out.with_columns(z_exprs)

    # 3. Basis Velocity & Acceleration
    vel_exprs = []
    for k in VELOCITY_LAGS:
        vel_exprs.append(
            (pl.col("basis_spread_bps") - pl.col("basis_spread_bps").shift(k)).fill_null(0.0).alias(f"basis_velocity_{k}0s")
        )
    out = out.with_columns(vel_exprs)

    acc_30s = pl.col("basis_velocity_30s") - pl.col("basis_velocity_30s").shift(3)
    out = out.with_columns(acc_30s.fill_null(0.0).alias("basis_acceleration_30s"))

    # 4. Funding Pressure & Divergence
    if out.schema["ts"] == pl.Datetime:
        curr_ts_ms = pl.col("ts").dt.epoch(time_unit="ms")
    else:
        curr_ts_ms = pl.col("ts")

    rem_ms = pl.max_horizontal(pl.lit(0), pl.col("_next_funding_time_ms") - curr_ts_ms)
    tau = pl.min_horizontal(pl.lit(1.0), rem_ms / (8.0 * 3600.0 * 1000.0))

    # When next_funding_time is not provided, use default tau=0.5
    f_pressure = pl.when(pl.col("_next_funding_time_ms") > 0)\
        .then(pl.col("_funding_rate") * (1.0 - tau) * 10000.0)\
        .otherwise(pl.col("_funding_rate") * 10000.0)

    f_ma8h = pl.col("_funding_rate").rolling_mean(2880, min_samples=1).fill_null(0.0)
    f_div = pl.col("basis_spread_bps") - (pl.col("_funding_rate") * 10000.0)

    out = out.with_columns([
        pl.col("_funding_rate").alias("funding_rate_raw"),
        f_ma8h.alias("funding_rate_ma_8h"),
        f_pressure.fill_null(0.0).alias("funding_pressure_bps"),
        f_div.fill_null(0.0).alias("funding_basis_divergence"),
        f_pressure.fill_null(0.0).alias("funding_rate_pressure"),  # alias
    ])

    drop_cols = ["_p_mid", "_s_mid", "_funding_rate", "_next_funding_time_ms"]
    if has_precomputed_basis:
        drop_cols.append("_pre_basis")
    out = out.drop([c for c in drop_cols if c in out.columns])

    return out.with_columns([
        pl.col(c).cast(pl.Float32).fill_null(0.0).fill_nan(0.0)
        for c in feature_cols if c in out.columns
    ])
