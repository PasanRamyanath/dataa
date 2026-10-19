"""
Multi-resolution context: brings the most recently CLOSED 1-minute and
5-minute bar's OHLC onto the primary (10s) grid via a backward as-of join —
a bar at time t only ever sees a higher-timeframe bar that has already
finished, never the one still forming around it.
"""
from __future__ import annotations

import polars as pl

from resample import asof_join_backward
from utils import safe_div


def add_multi_timeframe_features(
    primary: pl.DataFrame,
    bars_1m: pl.DataFrame | None,
    bars_5m: pl.DataFrame | None,
) -> pl.DataFrame:
    out = primary
    for bars, tf in ((bars_1m, "1m"), (bars_5m, "5m")):
        if bars is None or bars.is_empty():
            out = out.with_columns(pl.lit(None, dtype=pl.Float32).alias(f"htf_{tf}_close"))
            continue
        # Only join bars that have fully closed relative to `ts` — shift the
        # higher-timeframe close by one bar so the join can never pick the
        # still-forming bar containing the current primary-grid timestamp.
        htf = bars.select(
            "ts",
            pl.col("close").shift(1).alias(f"htf_{tf}_close"),
            pl.col("high").shift(1).alias(f"htf_{tf}_high"),
            pl.col("low").shift(1).alias(f"htf_{tf}_low"),
            (pl.col("close").shift(1).log() - pl.col("close").shift(2).log()).alias(f"htf_{tf}_return"),
        )
        out = asof_join_backward(out, htf)
        out = out.with_columns(
            [pl.col(c).forward_fill() for c in [f"htf_{tf}_close", f"htf_{tf}_high", f"htf_{tf}_low", f"htf_{tf}_return"]]
        )
        out = out.with_columns(
            safe_div(pl.col("close") - pl.col(f"htf_{tf}_close"), pl.col(f"htf_{tf}_close")).alias(f"htf_{tf}_deviation")
        )
    return out
