"""
Category 11: cross-exchange features from Bybit's BTCUSDT perpetual trades
(public.bybit.com, `trading/` tree).

Not part of the original plan's default scope (Bybit was set aside pending
"reconsideration for cross-exchange validation"), but the plan's own content
audit found Bybit's trade dump to be the cleanest data of anything sampled —
zero defects, vs. known issues in every Binance source. Two liquid perpetual
markets for the same underlying (Binance BTCUSDT vs. Bybit BTCUSDT) routinely
show brief price/order-flow divergence that mean-reverts as arbitrageurs
close the gap, and one venue's order flow can lead the other's by a handful
of bars — a genuinely independent, non-redundant signal source since it comes
from a disjoint pool of participants/liquidity rather than another derivative
of the same Binance trade tape every other feature family in this pipeline is
built from.

All joins are backward-only (see resample.asof_join_backward) and all
cross-venue lag features use an explicit `.shift(k)` on the already-lagged
Bybit bar close — never a centered or future-inclusive window — so this
family carries the same anti-leakage guarantee as the rest of the pipeline.
"""
from __future__ import annotations

import polars as pl

from resample import asof_join_backward, build_trade_bars
from utils import safe_div

LEAD_LAG_BAR_LAGS = [1, 2, 3, 5]   # 10s..50s at the primary 10s grid
DIVERGENCE_WINDOWS = [30, 180]


def build_bybit_bars(bybit_trades: pl.DataFrame, interval: str) -> pl.DataFrame:
    """Bybit's trading/ schema uses `size` for quantity and the already-fixed
    `is_buyer_maker` column (see parsers/bybit_parser.py) instead of
    Binance's `quantity` — otherwise identical in shape to a Binance trade
    stream, so this reuses the same bar builder."""
    return build_trade_bars(bybit_trades, interval, qty_col="size")


def add_cross_exchange_features(bars: pl.DataFrame, bybit_bars: pl.DataFrame | None) -> pl.DataFrame:
    if bybit_bars is None or bybit_bars.is_empty():
        cols = [pl.lit(None, dtype=pl.Float32).alias("cross_exch_spread_pct")]
        cols += [pl.lit(None, dtype=pl.Float32).alias(f"bybit_lead_return_lag_{k}") for k in LEAD_LAG_BAR_LAGS]
        cols += [pl.lit(None, dtype=pl.Float32).alias(f"cross_exch_ofi_divergence_{w}") for w in DIVERGENCE_WINDOWS]
        return bars.with_columns(cols)

    by = bybit_bars.select(
        "ts",
        pl.col("close").alias("bybit_close"),
        safe_div(pl.col("buy_volume") - pl.col("sell_volume"), pl.col("buy_volume") + pl.col("sell_volume"))
        .alias("bybit_taker_imbalance"),
    )
    out = asof_join_backward(bars, by)
    out = out.with_columns(pl.col("bybit_close").forward_fill(), pl.col("bybit_taker_imbalance").forward_fill())

    # Contemporaneous cross-venue basis: how far Binance trades from Bybit
    # right now — an arbitrage-pressure/divergence proxy independent of
    # anything derived from Binance's own order book or spot leg.
    out = out.with_columns(
        safe_div(pl.col("close") - pl.col("bybit_close"), pl.col("bybit_close")).alias("cross_exch_spread_pct")
    )

    # Explicit lagged Bybit return as a leading predictor of Binance's
    # current-bar return, mirroring the plan's spot-leads-futures hypothesis
    # but across venues instead of across spot/futures legs. Each lag is its
    # own column (not just a rolling correlation stat) so sequence/GBM
    # members can learn the relationship directly; `bybit_close` is already
    # as-of-backward joined so shifting it further only looks further into
    # the past, never forward.
    bybit_ret = pl.col("bybit_close").log().diff()
    out = out.with_columns([bybit_ret.shift(k).alias(f"bybit_lead_return_lag_{k}") for k in LEAD_LAG_BAR_LAGS])

    # Order-flow divergence: Bybit's taker buy/sell imbalance vs. Binance's
    # own (already computed upstream as `taker_buy_sell_imbalance` in
    # price_volume.add_volume_features, which must run before this in the
    # pipeline) — two independently-sourced order-flow signals disagreeing
    # is itself informative (e.g. one venue's aggressive buying not
    # confirmed by the other).
    if "taker_buy_sell_imbalance" in out.columns:
        out = out.with_columns(
            (pl.col("taker_buy_sell_imbalance") - pl.col("bybit_taker_imbalance")).alias("_ofi_divergence_raw")
        )
        for w in DIVERGENCE_WINDOWS:
            out = out.with_columns(
                pl.col("_ofi_divergence_raw").rolling_mean(window_size=w, min_samples=1).alias(f"cross_exch_ofi_divergence_{w}")
            )
        out = out.drop("_ofi_divergence_raw")
    else:
        out = out.with_columns([pl.lit(None, dtype=pl.Float32).alias(f"cross_exch_ofi_divergence_{w}") for w in DIVERGENCE_WINDOWS])

    return out.drop("bybit_taker_imbalance")


BASIS_DIVERGENCE_WINDOWS = [180, 2880]

_BYBIT_INDEX_OUTPUT_COLUMNS = (
    ["bybit_premium_index", "bybit_basis_pct", "cross_exch_basis_divergence"]
    + [f"cross_exch_basis_divergence_ma_{w}" for w in BASIS_DIVERGENCE_WINDOWS]
)


def add_bybit_index_features(bars: pl.DataFrame, bybit_premium_index: pl.DataFrame | None,
                              bybit_spot_index: pl.DataFrame | None) -> pl.DataFrame:
    """Bybit's own premium_index/ and spot_index/ dumps give an independent,
    second basis/premium axis alongside Binance's native
    futures_derivatives.add_basis_and_premium_features. Cross-exchange
    funding/premium divergence is an actively-monitored crowding/arbitrage
    signal (funding rates routinely diverge between Binance and Bybit; the
    divergence itself, not just each venue's own level, carries information).

    Both Bybit dumps are keyed under config.BYBIT_INDEX_SYMBOL (BTCUSD, the
    inverse-contract underlying — confirmed via sample audit that no BTCUSDT
    folder exists under either tree), distinct from BYBIT_SYMBOL used for
    trading/ above.

    Live-site-verified (2026-09-12): public.bybit.com's premium_index/BTCUSD/
    and spot_index/BTCUSD/ trees only ever contained 162 daily files each,
    spanning 2019-10-01..2020-03-10, and nothing since — the source has been
    dead site-wide for 6+ years, not merely stale for recent days. Every real
    build day this pipeline targets will therefore hit the
    `bybit_premium_index is None` branch below and this whole feature family
    (bybit_premium_index, bybit_basis_pct, cross_exch_basis_divergence, and
    its two moving averages) will be 100% null in the output. That's not a
    bug — the null-fallback below is exactly correct — but it means these 5
    columns currently carry no information for any day actually produced by
    this pipeline; kaggle_downloader.py deliberately no longer attempts these
    downloads (see its _daily_bybit_index_targets docstring). Left in place
    rather than deleted in case Bybit ever resumes publishing, or a training
    run wants to include historical 2019/2020 days where this data is real.

    `premium_index`'s `close` column is CONFIRMED (via direct sample
    inspection) to already be a tiny dimensionless premium rate (e.g.
    ~1e-4), directly analogous to Binance's own `premium_index` column
    (futures_derivatives.add_basis_and_premium_features) — NOT a price, so it
    must never be divided by anything. `spot_index`'s `close` IS a real BTC
    price. `bybit_basis_pct` is therefore built the same way Binance's
    `basis_pct` is: (own perp price - own index price) / own index price,
    using `bybit_close` (Bybit's own perp trade price, already backward
    as-of joined/forward-filled onto `bars` by add_cross_exchange_features,
    which must run before this in the pipeline) against the Bybit spot
    index, rather than mis-using the premium-rate dump for that role.
    """
    if bybit_premium_index is None or bybit_premium_index.is_empty():
        return bars.with_columns([pl.lit(None, dtype=pl.Float32).alias(c) for c in _BYBIT_INDEX_OUTPUT_COLUMNS])

    prem = bybit_premium_index.select("ts", pl.col("close").alias("bybit_premium_index"))
    out = asof_join_backward(bars, prem)
    out = out.with_columns(pl.col("bybit_premium_index").forward_fill())

    if bybit_spot_index is not None and not bybit_spot_index.is_empty() and "bybit_close" in out.columns:
        spot = bybit_spot_index.select("ts", pl.col("close").alias("_bybit_spot_index"))
        out = asof_join_backward(out, spot)
        out = out.with_columns(pl.col("_bybit_spot_index").forward_fill())
        out = out.with_columns(
            safe_div(pl.col("bybit_close") - pl.col("_bybit_spot_index"), pl.col("_bybit_spot_index")).alias("bybit_basis_pct")
        )
        out = out.drop("_bybit_spot_index")
    else:
        out = out.with_columns(pl.lit(None, dtype=pl.Float32).alias("bybit_basis_pct"))

    if "premium_index" in out.columns:
        out = out.with_columns(
            (pl.col("premium_index") - pl.col("bybit_premium_index")).alias("cross_exch_basis_divergence")
        )
        out = out.with_columns([
            pl.col("cross_exch_basis_divergence").rolling_mean(window_size=w, min_samples=1)
            .alias(f"cross_exch_basis_divergence_ma_{w}")
            for w in BASIS_DIVERGENCE_WINDOWS
        ])
    else:
        out = out.with_columns([pl.lit(None, dtype=pl.Float32).alias("cross_exch_basis_divergence")])
        out = out.with_columns([
            pl.lit(None, dtype=pl.Float32).alias(f"cross_exch_basis_divergence_ma_{w}")
            for w in BASIS_DIVERGENCE_WINDOWS
        ])

    return out
