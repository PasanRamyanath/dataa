"""
Category 1-2: price/return and volume features computed on a fixed-interval
OHLCV bar grid (output of resample.build_trade_bars).

All rolling ops use polars' default TRAILING window (right-aligned, ending at
the current row) — never `center=True` — so every feature at bar `t` depends
only on bars `<= t`. This is the pipeline's core anti-leakage guarantee and
must be preserved by any feature added here.

Feature-family evidence ranking (per project plan): order-flow imbalance and
VWAP deviation are the strongest documented signals; taker buy/sell ratio and
CVD are directionally useful but weaker/less consistently validated. Both
families are implemented here; downstream feature-selection can down-weight
the latter.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from utils import safe_div

RETURN_LAGS = [1, 3, 6, 12, 30, 60, 180]     # bars (at 10s: 10s..30min)
MOMENTUM_WINDOWS = [6, 30, 180, 720]
VOL_WINDOWS = [30, 90, 360]
LEVEL_WINDOWS = [30, 180, 720]
VOLUME_MA_WINDOWS = [30, 180, 720]


def add_price_return_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(pl.col("close").log().alias("_log_close"))

    return_exprs = [
        (pl.col("_log_close") - pl.col("_log_close").shift(lag)).alias(f"log_return_{lag}")
        for lag in RETURN_LAGS
    ]
    df = df.with_columns(return_exprs)

    momentum_exprs = [
        pl.col("log_return_1").rolling_sum(window_size=w).alias(f"momentum_{w}")
        for w in MOMENTUM_WINDOWS
    ]
    df = df.with_columns(momentum_exprs)

    vol_exprs = []
    for w in VOL_WINDOWS:
        vol_exprs.append(
            (pl.col("log_return_1").pow(2).rolling_sum(window_size=w)).sqrt().alias(f"realized_vol_{w}")
        )
    df = df.with_columns(vol_exprs)

    level_exprs = []
    for w in LEVEL_WINDOWS:
        roll_high = pl.col("high").rolling_max(window_size=w)
        roll_low = pl.col("low").rolling_min(window_size=w)
        level_exprs.append(safe_div(pl.col("close") - roll_low, roll_high - roll_low).alias(f"price_pct_of_range_{w}"))
        level_exprs.append(safe_div(pl.col("close") - roll_high, roll_high).alias(f"dist_from_high_{w}"))
        level_exprs.append(safe_div(pl.col("close") - roll_low, roll_low).alias(f"dist_from_low_{w}"))
    df = df.with_columns(level_exprs)

    df = df.with_columns(
        safe_div(pl.col("high") - pl.col("low"), pl.col("close")).alias("bar_range_pct"),
        safe_div(pl.col("close") - pl.col("open"), pl.col("open")).alias("bar_body_pct"),
    )
    return df.drop("_log_close")


def add_volume_features(df: pl.DataFrame) -> pl.DataFrame:
    ma_exprs = [pl.col("volume").rolling_mean(window_size=w).alias(f"volume_ma_{w}") for w in VOLUME_MA_WINDOWS]
    df = df.with_columns(ma_exprs)
    df = df.with_columns([
        safe_div(pl.col("volume"), pl.col(f"volume_ma_{w}")).alias(f"volume_ratio_{w}") for w in VOLUME_MA_WINDOWS
    ])

    df = df.with_columns(
        safe_div(pl.col("buy_volume") - pl.col("sell_volume"), pl.col("buy_volume") + pl.col("sell_volume"))
        .alias("taker_buy_sell_imbalance"),
        safe_div(pl.col("buy_volume"), pl.col("volume")).alias("taker_buy_ratio"),
    )

    # Cumulative volume delta, reset daily (assumes df is one calendar day;
    # the pipeline orchestrator is responsible for day-boundary segmentation).
    df = df.with_columns(pl.col("volume_delta").cum_sum().alias("cvd"))
    df = df.with_columns([
        pl.col("cvd").rolling_mean(window_size=w).alias(f"cvd_ma_{w}") for w in [30, 180]
    ])

    df = df.with_columns(
        safe_div(pl.col("max_trade_notional"), pl.col("notional_volume")).alias("large_trade_dominance"),
        safe_div(pl.col("trade_size_std"), safe_div(pl.col("volume"), pl.col("trade_count"))).alias("trade_size_cv"),
    )

    df = df.with_columns([
        safe_div(pl.col("close") - pl.col("vwap"), pl.col("vwap")).alias("vwap_deviation"),
    ])
    df = df.with_columns([
        pl.col("vwap_deviation").rolling_mean(window_size=w).alias(f"vwap_deviation_ma_{w}") for w in [30, 180]
    ])
    return df
