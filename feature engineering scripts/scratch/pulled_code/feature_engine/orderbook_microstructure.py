"""
Category 3-4: order-book / microstructure features.

Order-flow imbalance (from bookDepth snapshots) and the bid/ask spread are
the two highest-conviction feature families in the project plan's evidence
ranking. bookTicker is KNOWN STALE in the available samples (single file,
dated months before everything else) so it is intentionally NOT used as a
primary source here — bookDepth is the default, as recommended by the plan.

All "microstructure" estimators (Roll spread, Kyle's lambda, Amihud
illiquidity, a VPIN approximation, quote resilience) are computed from the
trade-bar stream only, so they degrade gracefully when book-depth snapshots
are unavailable for a given day/venue.
"""
from __future__ import annotations

import polars as pl

from resample import asof_join_backward
from utils import safe_div

OFI_WINDOWS = [30, 180]
MICROSTRUCTURE_WINDOWS = [30, 90, 360]

# Confirmed via direct inspection of a real bookDepth sample
# (BTCUSDT-bookDepth-2026-09-06.zip): `percentage` takes exactly these 12
# discrete signed values (~30s snapshot cadence), i.e. 6 symmetric levels.
BOOK_DEPTH_LEVELS = [0.2, 1.0, 2.0, 3.0, 4.0, 5.0]
_LEVEL_MEAN = sum(BOOK_DEPTH_LEVELS) / len(BOOK_DEPTH_LEVELS)
_LEVEL_SS = sum((l - _LEVEL_MEAN) ** 2 for l in BOOK_DEPTH_LEVELS)


def pivot_book_depth_snapshots(book_depth: pl.DataFrame) -> pl.DataFrame:
    """Turn the long (ts, percentage, depth, notional) bookDepth stream into
    one row per snapshot timestamp with bid-side / ask-side aggregates.

    Percentage < 0 is the bid side (below mid), > 0 is the ask side (above
    mid) per Binance's bookDepth convention; 0 does not occur in samples.
    """
    bd = book_depth.with_columns(
        pl.when(pl.col("percentage") < 0).then(pl.lit("bid")).otherwise(pl.lit("ask")).alias("side")
    )
    near = bd.filter(pl.col("percentage").abs() <= 1.0)
    wide = bd

    def _agg(frame: pl.DataFrame, suffix: str) -> pl.DataFrame:
        return frame.group_by(["ts", "side"]).agg(
            pl.col("depth").sum().alias("depth_sum"),
            pl.col("notional").sum().alias("notional_sum"),
        ).pivot(on="side", index="ts", values=["depth_sum", "notional_sum"]).rename({
            c: f"{c}_{suffix}" for c in ["depth_sum_bid", "depth_sum_ask", "notional_sum_bid", "notional_sum_ask"]
            if c not in ("ts",)
        })

    near_agg = _agg(near, "near")
    wide_agg = _agg(wide, "wide")
    out = near_agg.join(wide_agg, on="ts", how="full", coalesce=True).sort("ts")

    for suffix in ("near", "wide"):
        bid = pl.col(f"depth_sum_bid_{suffix}")
        ask = pl.col(f"depth_sum_ask_{suffix}")
        out = out.with_columns(
            safe_div(bid - ask, bid + ask).alias(f"order_flow_imbalance_{suffix}")
        )
    return out


def pivot_book_depth_levels(book_depth: pl.DataFrame) -> pl.DataFrame:
    """Per-level order-flow imbalance at each of bookDepth's 6 discrete
    percentage bands, kept separate rather than summed into the near/wide
    buckets above. This is what lets a downstream model see the *shape* of
    the imbalance across the book (e.g. bid-heavy at the touch but
    ask-heavy further out) instead of only a single collapsed number."""
    out = None
    for lvl in BOOK_DEPTH_LEVELS:
        bid = book_depth.filter((pl.col("percentage") + lvl).abs() < 1e-6).select("ts", pl.col("depth").alias("bid_depth"))
        ask = book_depth.filter((pl.col("percentage") - lvl).abs() < 1e-6).select("ts", pl.col("depth").alias("ask_depth"))
        lvl_df = bid.join(ask, on="ts", how="inner").with_columns(
            safe_div(pl.col("bid_depth") - pl.col("ask_depth"), pl.col("bid_depth") + pl.col("ask_depth")).alias(f"ofi_level_{lvl}")
        ).select("ts", f"ofi_level_{lvl}")
        out = lvl_df if out is None else out.join(lvl_df, on="ts", how="full", coalesce=True)
    return out.sort("ts")


def add_order_flow_imbalance_features(bars: pl.DataFrame, book_depth: pl.DataFrame | None) -> pl.DataFrame:
    level_cols = [f"ofi_level_{l}" for l in BOOK_DEPTH_LEVELS]
    if book_depth is None or book_depth.is_empty():
        cols = [
            pl.lit(None, dtype=pl.Float32).alias("order_flow_imbalance_near"),
            pl.lit(None, dtype=pl.Float32).alias("order_flow_imbalance_wide"),
        ]
        cols += [pl.lit(None, dtype=pl.Float32).alias(c) for c in level_cols]
        cols += [
            pl.lit(None, dtype=pl.Float32).alias("ofi_imbalance_decay_rate"),
            pl.lit(None, dtype=pl.Float32).alias("ofi_imbalance_decay_slope"),
        ]
        return bars.with_columns(cols)

    snapshots = pivot_book_depth_snapshots(book_depth)
    joined = asof_join_backward(bars, snapshots.select("ts", "order_flow_imbalance_near", "order_flow_imbalance_wide"))
    for w in OFI_WINDOWS:
        joined = joined.with_columns(
            pl.col("order_flow_imbalance_near").rolling_mean(window_size=w).alias(f"ofi_near_ma_{w}")
        )

    levels = pivot_book_depth_levels(book_depth)
    joined = asof_join_backward(joined, levels.select("ts", *level_cols))
    joined = joined.with_columns([pl.col(c).forward_fill() for c in level_cols])

    # "Imbalance decay rate": how the imbalance dissipates from the touch
    # (0.2%) out to the far level (5%) — the specific feature named in the
    # plan's §5.1 table, previously only present as a coarse 2-bucket
    # near/wide aggregate with no notion of decay across the book.
    joined = joined.with_columns(
        (pl.col(f"ofi_level_{BOOK_DEPTH_LEVELS[0]}") - pl.col(f"ofi_level_{BOOK_DEPTH_LEVELS[-1]}")).alias("ofi_imbalance_decay_rate")
    )
    # Linear-regression slope of imbalance vs. level (level is a fixed
    # constant per column, so this reduces to a per-row weighted sum —
    # Sum((level_i - mean_level) * ofi_i) / Sum((level_i - mean_level)^2)).
    slope_num = None
    for l in BOOK_DEPTH_LEVELS:
        term = (l - _LEVEL_MEAN) * pl.col(f"ofi_level_{l}")
        slope_num = term if slope_num is None else slope_num + term
    joined = joined.with_columns((slope_num / _LEVEL_SS).alias("ofi_imbalance_decay_slope"))
    return joined


def add_spread_features(bars: pl.DataFrame, book_ticker: pl.DataFrame | None) -> pl.DataFrame:
    """Spread/adverse-selection features. Falls back to a bar-implied proxy
    spread (high-low range) when bookTicker is unavailable, which is the
    expected case in production per the known-stale-sample finding above."""
    if book_ticker is None or book_ticker.is_empty():
        return bars.with_columns(
            safe_div(pl.col("high") - pl.col("low"), pl.col("close")).alias("proxy_spread_pct")
        )
    bt = book_ticker.with_columns(
        safe_div(pl.col("best_ask_price") - pl.col("best_bid_price"),
                 (pl.col("best_ask_price") + pl.col("best_bid_price")) / 2).alias("quoted_spread_pct"),
        safe_div(pl.col("best_bid_qty") - pl.col("best_ask_qty"),
                 pl.col("best_bid_qty") + pl.col("best_ask_qty")).alias("quote_size_imbalance"),
        ((pl.col("best_bid_price") * pl.col("best_ask_qty") + pl.col("best_ask_price") * pl.col("best_bid_qty"))
         / (pl.col("best_bid_qty") + pl.col("best_ask_qty"))).alias("microprice"),
    )
    out = asof_join_backward(bars, bt.select("ts", "quoted_spread_pct", "quote_size_imbalance", "microprice"))
    out = out.with_columns(safe_div(pl.col("close") - pl.col("microprice"), pl.col("microprice")).alias("microprice_deviation"))
    return out


def add_microstructure_estimators(bars: pl.DataFrame) -> pl.DataFrame:
    df = bars.with_columns(pl.col("close").log().diff().alias("_ret"))

    for w in MICROSTRUCTURE_WINDOWS:
        # Roll (1984) implied spread: 2*sqrt(-cov(r_t, r_t-1)) when the
        # autocovariance is negative (bid-ask bounce); null otherwise.
        cov = (
            (pl.col("_ret") - pl.col("_ret").rolling_mean(window_size=w))
            * (pl.col("_ret").shift(1) - pl.col("_ret").shift(1).rolling_mean(window_size=w))
        ).rolling_mean(window_size=w)
        df = df.with_columns(
            pl.when(cov < 0).then((-cov).sqrt() * 2).otherwise(None).alias(f"roll_spread_{w}")
        )

        # Kyle's lambda: price-impact-per-unit-signed-volume, estimated as
        # cov(ret, signed_volume) / var(signed_volume) over a trailing window.
        signed_vol = pl.col("volume_delta")
        cov_rv = (
            (pl.col("_ret") - pl.col("_ret").rolling_mean(window_size=w))
            * (signed_vol - signed_vol.rolling_mean(window_size=w))
        ).rolling_mean(window_size=w)
        var_v = signed_vol.rolling_std(window_size=w).pow(2)
        df = df.with_columns(safe_div(cov_rv, var_v).alias(f"kyle_lambda_{w}"))

        # Amihud illiquidity: mean(|return| / dollar volume) over the window.
        df = df.with_columns(
            safe_div(pl.col("_ret").abs(), pl.col("notional_volume"))
            .rolling_mean(window_size=w).alias(f"amihud_illiquidity_{w}")
        )

        # VPIN-style volume-synchronized order imbalance: rolling mean of
        # |buy-sell| / total volume, a simplified proxy for the full
        # volume-bucketed VPIN (which requires bulk-volume classification
        # over fixed-volume buckets rather than fixed-time bars).
        df = df.with_columns(
            safe_div((pl.col("buy_volume") - pl.col("sell_volume")).abs(), pl.col("volume"))
            .rolling_mean(window_size=w).alias(f"vpin_proxy_{w}")
        )

    df = df.with_columns(
        safe_div(pl.col("close") - pl.col("close").shift(1), pl.col("volume_delta").abs() + 1e-9)
        .alias("quote_resilience")
    )
    return df.drop("_ret")
