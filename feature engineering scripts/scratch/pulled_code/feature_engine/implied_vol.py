"""
Category 12: implied-volatility features from Binance's BVOLIndex
(option/daily/BVOLIndex/) — BTC's annualized implied-volatility index, a
forward-looking measure structurally distinct from realized volatility
computed from price history alone. Web research confirms the spread between
implied and realized vol (the "variance risk premium") is a well-documented
directional/regime signal, and IV level/term is itself a standard input to
options-adjacent crypto strategies.

Must run AFTER feature_engine.price_volume.add_price_return_features, which
produces `realized_vol_90` (the stdev of 10s log-returns accumulated over a
90-bar/15-minute trailing window, NOT annualized) — this module annualizes it
to Binance's IV percentage scale to build `iv_realized_vol_spread`.

Like every other optional-source feature family, an unavailable day (no
BVOLIndex file published, or Bybit BVOL predates the sample window) falls
back to an all-None float32 schema so cross-day concatenation never breaks.
"""
from __future__ import annotations

import math

import polars as pl

from config import PRIMARY_INTERVAL_SECONDS
from resample import asof_join_backward

# Must match feature_engine.price_volume.VOL_WINDOWS[1] (realized_vol_90).
REALIZED_VOL_WINDOW_BARS = 90
SECONDS_PER_YEAR = 365 * 24 * 3600
BARS_PER_YEAR = SECONDS_PER_YEAR / PRIMARY_INTERVAL_SECONDS
REALIZED_VOL_ANNUALIZATION_PCT = math.sqrt(BARS_PER_YEAR / REALIZED_VOL_WINDOW_BARS) * 100

IV_MA_WINDOWS = [30, 180, 720]

_OUTPUT_COLUMNS = (
    ["implied_vol", "implied_vol_change_1", "iv_realized_vol_spread"]
    + [f"implied_vol_ma_{w}" for w in IV_MA_WINDOWS]
)


def add_implied_volatility_features(bars: pl.DataFrame, bvol_index: pl.DataFrame | None) -> pl.DataFrame:
    if bvol_index is None or bvol_index.is_empty():
        return bars.with_columns([pl.lit(None, dtype=pl.Float32).alias(c) for c in _OUTPUT_COLUMNS])

    iv = bvol_index.select("ts", pl.col("index_value").alias("implied_vol"))
    out = asof_join_backward(bars, iv)
    out = out.with_columns(pl.col("implied_vol").forward_fill())

    out = out.with_columns(pl.col("implied_vol").diff().alias("implied_vol_change_1"))
    out = out.with_columns([
        pl.col("implied_vol").rolling_mean(window_size=w, min_samples=1).alias(f"implied_vol_ma_{w}")
        for w in IV_MA_WINDOWS
    ])

    if "realized_vol_90" in out.columns:
        annualized_realized_vol_pct = pl.col("realized_vol_90") * REALIZED_VOL_ANNUALIZATION_PCT
        out = out.with_columns((pl.col("implied_vol") - annualized_realized_vol_pct).alias("iv_realized_vol_spread"))
    else:
        out = out.with_columns(pl.lit(None, dtype=pl.Float32).alias("iv_realized_vol_spread"))

    return out
