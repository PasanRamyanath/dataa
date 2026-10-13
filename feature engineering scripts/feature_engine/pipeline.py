"""
Orchestrates the full feature-computation pass for one symbol/day: takes the
primary bar grid (built from raw futures trades) plus whatever auxiliary
sources were available that day, and runs every feature-family module in a
fixed order. Order matters: later modules assume columns computed by earlier
ones exist (e.g. technical/statistical modules consume `log_return_1` and
`realized_vol_90` produced by price_volume).
"""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from feature_engine.cross_exchange import add_bybit_index_features, add_cross_exchange_features
from feature_engine.futures_derivatives import (
    add_basis_and_premium_features,
    add_funding_features,
    add_funding_oi_interaction_features,
    add_open_interest_features,
    add_spot_futures_lead_lag_features,
)
from feature_engine.implied_vol import add_implied_volatility_features
from feature_engine.lead_lag_liquidation import (
    add_breakout_support_resistance_features,
    add_liquidation_features,
)
from feature_engine.orderbook_microstructure import (
    add_microstructure_estimators,
    add_order_flow_imbalance_features,
    add_spread_features,
)
from feature_engine.multi_timeframe import add_multi_timeframe_features
from feature_engine.price_volume import add_price_return_features, add_volume_features
from feature_engine.technical_statistical import (
    add_statistical_regime_features,
    add_technical_indicators,
)
from feature_engine.temporal import add_temporal_features
from feature_engine.term_structure import add_term_structure_features
from utils import clean_inf_nan, downcast_floats_to_f32


@dataclass
class DaySources:
    """Everything available for one symbol/day; any field may be None if the
    corresponding endpoint/file was missing, zero-byte, or unpublished."""
    futures_bars: pl.DataFrame                         # required: primary bar grid
    book_depth: pl.DataFrame | None = None
    book_ticker: pl.DataFrame | None = None
    funding_rate: pl.DataFrame | None = None
    metrics: pl.DataFrame | None = None
    index_klines: pl.DataFrame | None = None
    mark_klines: pl.DataFrame | None = None
    premium_klines: pl.DataFrame | None = None
    spot_bars: pl.DataFrame | None = None
    liquidations: pl.DataFrame | None = None
    bars_1m: pl.DataFrame | None = None
    bars_5m: pl.DataFrame | None = None
    bybit_bars: pl.DataFrame | None = None
    bvol_index: pl.DataFrame | None = None
    bybit_premium_index: pl.DataFrame | None = None
    bybit_spot_index: pl.DataFrame | None = None
    front_quarter_klines: pl.DataFrame | None = None
    back_quarter_klines: pl.DataFrame | None = None


def compute_all_features(sources: DaySources) -> pl.DataFrame:
    df = sources.futures_bars

    df = add_price_return_features(df)
    df = add_volume_features(df)

    df = add_order_flow_imbalance_features(df, sources.book_depth)
    df = add_spread_features(df, sources.book_ticker)
    df = add_microstructure_estimators(df)

    df = add_funding_features(df, sources.funding_rate)
    df = add_open_interest_features(df, sources.metrics)
    df = add_funding_oi_interaction_features(df)
    df = add_basis_and_premium_features(df, sources.index_klines, sources.mark_klines, sources.premium_klines)
    df = add_spot_futures_lead_lag_features(df, sources.spot_bars)
    df = add_cross_exchange_features(df, sources.bybit_bars)  # needs taker_buy_sell_imbalance from add_volume_features above
    df = add_bybit_index_features(df, sources.bybit_premium_index, sources.bybit_spot_index)  # needs bybit_close from add_cross_exchange_features and premium_index from add_basis_and_premium_features above
    df = add_implied_volatility_features(df, sources.bvol_index)  # needs realized_vol_90 from add_price_return_features above
    df = add_term_structure_features(df, sources.front_quarter_klines, sources.back_quarter_klines)

    df = add_technical_indicators(df)
    df = add_statistical_regime_features(df)

    df = add_breakout_support_resistance_features(df)
    df = add_liquidation_features(df, sources.liquidations)
    df = add_multi_timeframe_features(df, sources.bars_1m, sources.bars_5m)

    df = add_temporal_features(df)

    df = clean_inf_nan(df)
    df = downcast_floats_to_f32(df)
    return df


def feature_manifest(df: pl.DataFrame) -> dict:
    """Lightweight audit of the produced feature set — column count, dtypes,
    and per-column null rate, useful for the HF dataset card / sanity checks."""
    null_rates = {c: float(df.select(pl.col(c).is_null().mean()).item()) for c in df.columns}
    return {
        "n_columns": df.width,
        "n_rows": df.height,
        "columns": df.columns,
        "null_rates": null_rates,
    }
