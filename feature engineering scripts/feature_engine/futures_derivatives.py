"""
Category 5-6: futures-specific features (funding, open interest, basis/
premium) and positioning/sentiment features (long-short ratios, taker
volume). Per the plan's evidence ranking, funding-rate/OI and basis/premium
are mid-strength signals; long-short ratios and taker-volume ratios are
weaker/less consistently validated and are included as auxiliary features
rather than primary ones.

All lower-frequency sources (funding rate: every 8h; metrics: ~5min;
klines-family: whatever interval was downloaded) are joined onto the bar
grid with a BACKWARD as-of join only — a bar at time t only ever sees the
most recently published value at or before t, never a future one.
"""
from __future__ import annotations

import polars as pl

from resample import asof_join_backward
from utils import safe_div

FUNDING_MA_WINDOWS_BARS = {"8h": 2880, "24h": 8640, "7d": 60480}  # @ 10s bars
OI_ROC_WINDOWS = [30, 180, 720]


def add_funding_features(bars: pl.DataFrame, funding_rate: pl.DataFrame | None) -> pl.DataFrame:
    if funding_rate is None or funding_rate.is_empty():
        return bars.with_columns(pl.lit(None, dtype=pl.Float32).alias("funding_rate"))
    fr = funding_rate.select("ts", pl.col("last_funding_rate").alias("funding_rate"))
    out = asof_join_backward(bars, fr)
    out = out.with_columns(pl.col("funding_rate").forward_fill())
    for name, w in FUNDING_MA_WINDOWS_BARS.items():
        out = out.with_columns(pl.col("funding_rate").rolling_mean(window_size=w, min_samples=1).alias(f"funding_rate_ma_{name}"))
    out = out.with_columns(
        (pl.col("funding_rate") - pl.col("funding_rate").shift(1)).alias("funding_rate_change"),
        pl.col("funding_rate").rolling_std(window_size=FUNDING_MA_WINDOWS_BARS["24h"], min_samples=1).alias("funding_rate_vol_24h"),
    )
    return out


def add_open_interest_features(bars: pl.DataFrame, metrics: pl.DataFrame | None) -> pl.DataFrame:
    if metrics is None or metrics.is_empty():
        return bars.with_columns([
            pl.lit(None, dtype=pl.Float32).alias("open_interest"),
            pl.lit(None, dtype=pl.Float32).alias("open_interest_value"),
        ])
    m = metrics.select(
        "ts",
        pl.col("sum_open_interest").alias("open_interest"),
        pl.col("sum_open_interest_value").alias("open_interest_value"),
        pl.col("count_toptrader_long_short_ratio").alias("toptrader_count_ls_ratio"),
        pl.col("sum_toptrader_long_short_ratio").alias("toptrader_sum_ls_ratio"),
        pl.col("count_long_short_ratio").alias("all_account_ls_ratio"),
        pl.col("sum_taker_long_short_vol_ratio").alias("taker_ls_vol_ratio"),
    )
    out = asof_join_backward(bars, m)
    ffill_cols = ["open_interest", "open_interest_value", "toptrader_count_ls_ratio",
                  "toptrader_sum_ls_ratio", "all_account_ls_ratio", "taker_ls_vol_ratio"]
    out = out.with_columns([pl.col(c).forward_fill() for c in ffill_cols])

    for w in OI_ROC_WINDOWS:
        out = out.with_columns(
            safe_div(pl.col("open_interest") - pl.col("open_interest").shift(w), pl.col("open_interest").shift(w))
            .alias(f"oi_roc_{w}")
        )
    out = out.with_columns(
        safe_div(pl.col("open_interest") - pl.col("open_interest").shift(1), pl.col("open_interest").shift(1))
        .alias("oi_change_pct"),
        safe_div(pl.col("volume"), pl.col("open_interest")).alias("volume_to_oi_ratio"),
    )
    return out


def add_basis_and_premium_features(bars: pl.DataFrame, index_klines: pl.DataFrame | None,
                                    mark_klines: pl.DataFrame | None,
                                    premium_klines: pl.DataFrame | None) -> pl.DataFrame:
    out = bars
    if index_klines is not None and not index_klines.is_empty():
        idx = index_klines.select("ts", pl.col("close").alias("index_price"))
        out = asof_join_backward(out, idx)
        out = out.with_columns(pl.col("index_price").forward_fill())
        out = out.with_columns(
            safe_div(pl.col("close") - pl.col("index_price"), pl.col("index_price")).alias("basis_pct")
        )
        out = out.with_columns([
            pl.col("basis_pct").rolling_mean(window_size=w, min_samples=1).alias(f"basis_pct_ma_{w}")
            for w in [180, 720]
        ])
    else:
        out = out.with_columns(pl.lit(None, dtype=pl.Float32).alias("basis_pct"))

    if mark_klines is not None and not mark_klines.is_empty():
        mk = mark_klines.select("ts", pl.col("close").alias("mark_price"))
        out = asof_join_backward(out, mk)
        out = out.with_columns(pl.col("mark_price").forward_fill())
        out = out.with_columns(
            safe_div(pl.col("close") - pl.col("mark_price"), pl.col("mark_price")).alias("mark_price_deviation")
        )
    else:
        out = out.with_columns(pl.lit(None, dtype=pl.Float32).alias("mark_price_deviation"))

    if premium_klines is not None and not premium_klines.is_empty():
        pk = premium_klines.select("ts", pl.col("close").alias("premium_index"))
        out = asof_join_backward(out, pk)
        out = out.with_columns(pl.col("premium_index").forward_fill())
        out = out.with_columns([
            pl.col("premium_index").rolling_mean(window_size=w, min_samples=1).alias(f"premium_index_ma_{w}")
            for w in [180, 2880]
        ])
    else:
        out = out.with_columns(pl.lit(None, dtype=pl.Float32).alias("premium_index"))

    return out


SPOT_LEAD_LAG_BARS = [1, 3, 6, 30]  # 10s..300s @ 10s bars


def add_spot_futures_lead_lag_features(bars: pl.DataFrame, spot_bars: pl.DataFrame | None) -> pl.DataFrame:
    """Cross-market lead-lag: spot-vs-futures return spread, rolling
    divergence, and explicit lagged spot returns as leading-predictor
    columns. Requires spot bars built on the same bar grid/interval.

    The plan (§5.1) names spot return over the last 100-500ms as a candidate
    leading feature for short-horizon futures return, to be verified
    empirically. Our primary grid is 10s bars (config.PRIMARY_INTERVAL), two
    orders of magnitude coarser than that window, so `spot_lead_return_lag_1`
    (one bar back, 10s) is the finest resolution available here rather than
    a literal 100-500ms feature; the wider lags let a model pick whichever
    horizon actually carries signal at this resolution instead of assuming
    it up front. Each lag is shifted from `spot_close`, which is itself only
    ever backward as-of joined/forward-filled, so no lag can see a future
    spot print relative to the current futures bar.
    """
    if spot_bars is None or spot_bars.is_empty():
        cols = [pl.lit(None, dtype=pl.Float32).alias("spot_futures_spread_pct")]
        cols += [pl.lit(None, dtype=pl.Float32).alias(f"spot_lead_return_lag_{k}") for k in SPOT_LEAD_LAG_BARS]
        return bars.with_columns(cols)
    spot = spot_bars.select("ts", pl.col("close").alias("spot_close"))
    out = asof_join_backward(bars, spot)
    out = out.with_columns(pl.col("spot_close").forward_fill())
    out = out.with_columns(
        safe_div(pl.col("close") - pl.col("spot_close"), pl.col("spot_close")).alias("spot_futures_spread_pct"),
        (pl.col("close").log().diff() - pl.col("spot_close").log().diff()).alias("_lead_lag_ret_diff"),
    )
    for w in [30, 180]:
        out = out.with_columns(
            pl.col("_lead_lag_ret_diff").rolling_std(window_size=w, min_samples=1).alias(f"spot_futures_ret_divergence_{w}")
        )
    spot_ret = pl.col("spot_close").log().diff()
    out = out.with_columns([spot_ret.shift(k).alias(f"spot_lead_return_lag_{k}") for k in SPOT_LEAD_LAG_BARS])
    return out.drop("_lead_lag_ret_diff")


def add_funding_oi_interaction_features(bars: pl.DataFrame) -> pl.DataFrame:
    """Funding-rate x open-interest interaction: a "crowding/squeeze proxy",
    named explicitly in the plan's §5.1 ranked feature-family table but not
    previously implemented (funding and OI were only ever added as separate,
    non-interacting features). Persistently elevated funding in one
    direction combined with *rising* OI indicates a crowded directional
    position building up (e.g. longs paying shorts while OI keeps growing
    means more leveraged longs entering) -- the setup that historically
    precedes a squeeze/liquidation cascade in the opposite direction.
    Neither `funding_rate` alone nor `oi_roc_*` alone captures this; it only
    shows up in their product/co-movement.

    Requires `add_funding_features` and `add_open_interest_features` to have
    already run on `bars` (both always run earlier in
    feature_engine.pipeline.compute_all_features). Their rolling-MA/ROC
    columns are only added on the populated-data branch, not the
    source-unavailable branch, so this always emits the same two output
    columns (null when either input was unavailable that day) to keep the
    schema identical across days for concatenation.
    """
    have_funding = "funding_rate_ma_8h" in bars.columns
    have_oi = "oi_roc_180" in bars.columns and "oi_roc_720" in bars.columns
    if not (have_funding and have_oi):
        return bars.with_columns([
            pl.lit(None, dtype=pl.Float32).alias("funding_oi_crowding_180"),
            pl.lit(None, dtype=pl.Float32).alias("funding_oi_crowding_720"),
        ])
    return bars.with_columns(
        (pl.col("funding_rate") * pl.col("oi_roc_180")).alias("funding_oi_crowding_180"),
        (pl.col("funding_rate_ma_8h") * pl.col("oi_roc_720")).alias("funding_oi_crowding_720"),
    )
