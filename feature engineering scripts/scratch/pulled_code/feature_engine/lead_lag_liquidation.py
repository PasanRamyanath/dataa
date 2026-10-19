"""
Breakout / support-resistance signals and liquidation heuristics.

Liquidation heuristics are the weakest-evidence feature family in the plan's
ranking (liquidationSnapshot is cm-only, sparse, and the sample data is
known to contain 50% duplicate rows — always dedup before use, which
parsers.binance_parser.read_liquidation_snapshot already does). Treated
here as low-confidence auxiliary signals, not primary features.
"""
from __future__ import annotations

import polars as pl

from utils import safe_div

BREAKOUT_WINDOWS = [180, 720]
LIQUIDATION_WINDOWS_BARS = [360, 2880]  # ~1hr, ~8hr @ 10s bars


def add_breakout_support_resistance_features(df: pl.DataFrame) -> pl.DataFrame:
    for w in BREAKOUT_WINDOWS:
        roll_high = pl.col("high").rolling_max(window_size=w).shift(1)
        roll_low = pl.col("low").rolling_min(window_size=w).shift(1)
        df = df.with_columns(
            (pl.col("close") > roll_high).cast(pl.Int8).alias(f"breakout_up_{w}"),
            (pl.col("close") < roll_low).cast(pl.Int8).alias(f"breakout_down_{w}"),
            safe_div(pl.col("close") - roll_high, pl.col("close")).alias(f"resistance_distance_{w}"),
            safe_div(pl.col("close") - roll_low, pl.col("close")).alias(f"support_distance_{w}"),
        )
    return df


def add_liquidation_features(bars: pl.DataFrame, liquidations: pl.DataFrame | None) -> pl.DataFrame:
    if liquidations is None or liquidations.is_empty():
        # Must match the populated-path schema exactly (raw + both windows)
        # so days with and without liquidation data concatenate cleanly.
        # Null (not 0.0), matching every other optional-source module's
        # convention (funding, OI, basis, cross-exchange, term structure):
        # 0.0 would mean "confirmed zero liquidations", which is a real,
        # different signal from "source unavailable this day" — a
        # distinction that matters a lot in practice since liquidationSnapshot
        # is confirmed discontinued platform-wide after 2024-10-14, so this
        # branch is now the common case for most of the dataset's date range,
        # not a rare edge case.
        return bars.with_columns([
            pl.lit(None, dtype=pl.Float32).alias("liquidation_notional"),
            pl.lit(None, dtype=pl.Float32).alias("liquidation_notional_1h"),
            pl.lit(None, dtype=pl.Float32).alias("liquidation_side_imbalance_1h"),
            pl.lit(None, dtype=pl.Float32).alias("liquidation_notional_8h"),
            pl.lit(None, dtype=pl.Float32).alias("liquidation_side_imbalance_8h"),
        ])

    liq = liquidations.with_columns(
        (pl.col("average_price") * pl.col("last_fill_quantity")).alias("notional"),
        pl.when(pl.col("side") == "SELL").then(pl.col("average_price") * pl.col("last_fill_quantity")).otherwise(0.0).alias("sell_notional"),
        pl.when(pl.col("side") == "BUY").then(pl.col("average_price") * pl.col("last_fill_quantity")).otherwise(0.0).alias("buy_notional"),
    )
    per_bar = liq.with_columns(pl.from_epoch(pl.col("ts"), time_unit="ns").alias("ts")).sort("ts").group_by_dynamic(
        "ts", every="10s", closed="left", label="left"
    ).agg(
        pl.col("notional").sum().alias("liquidation_notional"),
        pl.col("buy_notional").sum().alias("liquidation_buy_notional"),
        pl.col("sell_notional").sum().alias("liquidation_sell_notional"),
    )

    # Exact-match left join, NOT an as-of join: `per_bar` only has rows for
    # 10s bins that actually had a liquidation, and both grids are binned
    # from the same epoch-aligned origin. An as-of-backward join would carry
    # each liquidation's notional forward into every later bar indefinitely
    # (treating a sparse per-bin event like a continuously-updating state
    # variable), double-counting it in every subsequent rolling window
    # instead of only the bin it happened in.
    out = bars.join(per_bar, on="ts", how="left")
    out = out.with_columns([
        pl.col("liquidation_notional").fill_null(0.0),
        pl.col("liquidation_buy_notional").fill_null(0.0),
        pl.col("liquidation_sell_notional").fill_null(0.0),
    ])
    for w in LIQUIDATION_WINDOWS_BARS:
        label = "1h" if w == 360 else "8h"
        out = out.with_columns(
            pl.col("liquidation_notional").rolling_sum(window_size=w, min_samples=1).alias(f"liquidation_notional_{label}")
        )
        buy_roll = pl.col("liquidation_buy_notional").rolling_sum(window_size=w, min_samples=1)
        sell_roll = pl.col("liquidation_sell_notional").rolling_sum(window_size=w, min_samples=1)
        out = out.with_columns(
            safe_div(buy_roll - sell_roll, buy_roll + sell_roll).alias(f"liquidation_side_imbalance_{label}")
        )
    return out.drop("liquidation_buy_notional", "liquidation_sell_notional")
