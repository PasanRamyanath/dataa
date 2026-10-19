"""
Category 13: futures term-structure (calendar-spread) features from the
front/back quarterly delivery-futures contracts (data/futures/{um,cm}/daily/
markPriceKlines/{BASE}_{YYMMDD}/) — see contracts.py for how the two dated
contract symbols active on a given day are derived.

Web research confirms calendar-spread TREND (not just its static level) has
documented statistical significance for predicting the direction/magnitude
of a leading futures contract's return, and that BTC futures exhibit
structural contango driven by sentiment/momentum rather than pure
cost-of-carry (unlike traditional commodities) — so this module emits both
levels and an explicit rate-of-change/momentum column on the front-quarter
basis, not just the static spread.

Each quarterly basis is computed the same way as the perpetual's own
`basis_pct` in futures_derivatives.add_basis_and_premium_features: (quarterly
mark price - perpetual close) / perpetual close, using the PERPETUAL's own
close as the underlying reference price (there is no separate "index price"
series for a specific quarterly contract in this data) — a standard
practice-driven proxy since spot/perp and index price track closely.

Null-schema fallback applies whenever either quarterly contract's
markPriceKlines is missing for the day (e.g. before quarterly futures
existed for this base, or a downloader miss) so cross-day concatenation
never breaks schema consistency.
"""
from __future__ import annotations

import polars as pl

from resample import asof_join_backward
from utils import safe_div

BASIS_MOMENTUM_WINDOWS = [180, 720]  # 30min / 2h @ 10s bars

_OUTPUT_COLUMNS = (
    ["calendar_basis_front_pct", "calendar_basis_back_pct", "term_structure_slope"]
    + [f"calendar_basis_front_momentum_{w}" for w in BASIS_MOMENTUM_WINDOWS]
)


def add_term_structure_features(bars: pl.DataFrame, front_quarter_klines: pl.DataFrame | None,
                                 back_quarter_klines: pl.DataFrame | None) -> pl.DataFrame:
    if front_quarter_klines is None or front_quarter_klines.is_empty() or \
       back_quarter_klines is None or back_quarter_klines.is_empty():
        return bars.with_columns([pl.lit(None, dtype=pl.Float32).alias(c) for c in _OUTPUT_COLUMNS])

    front = front_quarter_klines.select("ts", pl.col("close").alias("_front_quarter_price"))
    back = back_quarter_klines.select("ts", pl.col("close").alias("_back_quarter_price"))

    out = asof_join_backward(bars, front)
    out = out.with_columns(pl.col("_front_quarter_price").forward_fill())
    out = asof_join_backward(out, back)
    out = out.with_columns(pl.col("_back_quarter_price").forward_fill())

    out = out.with_columns(
        safe_div(pl.col("_front_quarter_price") - pl.col("close"), pl.col("close")).alias("calendar_basis_front_pct"),
        safe_div(pl.col("_back_quarter_price") - pl.col("close"), pl.col("close")).alias("calendar_basis_back_pct"),
    )
    out = out.with_columns(
        (pl.col("calendar_basis_back_pct") - pl.col("calendar_basis_front_pct")).alias("term_structure_slope")
    )
    out = out.with_columns([
        pl.col("calendar_basis_front_pct").diff(n=w).alias(f"calendar_basis_front_momentum_{w}")
        for w in BASIS_MOMENTUM_WINDOWS
    ])

    return out.drop("_front_quarter_price", "_back_quarter_price")
