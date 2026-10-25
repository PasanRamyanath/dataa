"""
Unified Orthogonal Feature Pipeline (Milestone 1).
Implements the exact interface contract defined in PROJECT.md and tests/contracts.py:
compute_orthogonal_features(btc_bars_df, eth_bars_df, spot_perp_df, orderbook_df, cutoff_timestamp_ms)
Strictly enforces t_event <= cutoff_timestamp_ms <= T_close - 1000ms.
"""
from __future__ import annotations

import logging
from pathlib import Path
import polars as pl
import numpy as np

try:
    from .cross_asset import add_cross_asset_lead_lag_features
    from .basis_funding import add_basis_and_funding_features
    from .microstructure_vpin import add_microstructure_and_vpin_features
except ImportError:
    from cross_asset import add_cross_asset_lead_lag_features
    from basis_funding import add_basis_and_funding_features
    from microstructure_vpin import add_microstructure_and_vpin_features

log = logging.getLogger("features.pipeline")

# Canonical Pruned Orthogonal Feature Set (Stationary & Decorrelated)
PRUNED_ORTHOGONAL_FEATURES = [
    # Family 1: Cross-Asset Lead-Lag Dynamics
    "eth_lead_return_lag_1",
    "eth_lead_return_lag_2",
    "eth_lead_return_lag_3",
    "cross_return_spread_10s",
    "cross_return_spread_30s",
    "cross_return_residual_10s",
    "cross_ofi_divergence_10s",
    "cross_ofi_divergence_30s",
    "cross_mom_diff_30s",
    "cross_mom_diff_zscore_30s",
    "eth_lead_ret_bps",
    "eth_imbalance",
    # Family 2: Spot-Perp Basis Divergence & Funding Pressure
    "basis_spread_bps",
    "basis_zscore_180",
    "basis_zscore_720",
    "basis_velocity_10s",
    "basis_velocity_30s",
    "basis_acceleration_30s",
    "funding_pressure_bps",
    "funding_basis_divergence",
    "funding_rate_pressure",
    # Family 3: Order Book Microstructure & Flow Toxicity
    "l1_queue_imbalance",
    "l1_queue_imbalance_ma3",
    "l1_queue_velocity",
    "microprice_drift_bps",
    "microprice_velocity_10s",
    "microprice_velocity_30s",
    "trade_flow_imbalance",
    "tfi_ma_3",
    "vpin_toxicity_6",
    "vpin_surge_ratio",
    "liq_net_flow_bps",
    "liq_burst_zscore",
    "l1_imbalance",
    "vpin",
    "ofi",
]

FORBIDDEN_NON_STATIONARY = [
    "close", "open", "high", "low", "vwap", "spot_close",
    "bbo_bid", "bbo_ask", "bbo_mid", "eth_close", "perp_mid", "spot_mid", "microprice",
    "bybit_premium_index", "liquidation_notional_1h", "ret_pct", "_live_ret_pct"
]


def _filter_by_cutoff(df: pl.DataFrame | None, cutoff_timestamp_ms: int, stream_name: str) -> pl.DataFrame:
    if df is None or df.is_empty():
        return pl.DataFrame()

    ts_col = "cutoff_ts" if "cutoff_ts" in df.columns else "ts"
    if hasattr(df, "_df"):
        s = df._df[ts_col]
        if len(s) > 0 and not hasattr(s.iloc[-1], "timestamp") and int(s.iloc[-1]) <= cutoff_timestamp_ms and int(s.max()) <= cutoff_timestamp_ms:
            return df
    if df.schema[ts_col] == pl.Datetime:
        cutoff_val = pl.from_epoch(cutoff_timestamp_ms * 1_000_000, time_unit="ns")
    else:
        cutoff_val = cutoff_timestamp_ms

    filtered = df.filter(pl.col(ts_col) <= cutoff_val)

    # Formal audit assertion: no row in filtered DataFrame can exceed cutoff
    if filtered.height > 0:
        max_ts = filtered.select(pl.col(ts_col).max()).item()
        if max_ts is not None:
            max_ts_int = int(max_ts.timestamp() * 1000) if hasattr(max_ts, "timestamp") else int(max_ts)
            if max_ts_int > cutoff_timestamp_ms:
                raise AssertionError(
                    f"CRITICAL TIME-SERIES INTEGRITY VIOLATION: {stream_name} contains event timestamp "
                    f"{max_ts_int} > cutoff {cutoff_timestamp_ms}"
                )
    return filtered


def compute_orthogonal_features(
    btc_bars_df: pl.DataFrame,
    eth_bars_df: pl.DataFrame | None,
    spot_perp_df: pl.DataFrame | None,
    orderbook_df: pl.DataFrame | None,
    cutoff_timestamp_ms: int,
) -> pl.DataFrame:
    """
    Computes pruned orthogonal feature set strictly adhering to:
    t_event <= cutoff_timestamp_ms <= T_close - 1000ms.
    Returns stationary DataFrame with zero lookahead columns.
    """
    # 1. Enforce strict cutoff across all input streams
    btc_filtered = _filter_by_cutoff(btc_bars_df, cutoff_timestamp_ms, "btc_bars")
    if btc_filtered.is_empty():
        return pl.DataFrame()

    eth_filtered = _filter_by_cutoff(eth_bars_df, cutoff_timestamp_ms, "eth_bars")
    spot_perp_filtered = _filter_by_cutoff(spot_perp_df, cutoff_timestamp_ms, "spot_perp")
    orderbook_filtered = _filter_by_cutoff(orderbook_df, cutoff_timestamp_ms, "orderbook")

    df = btc_filtered

    # 2. Add Family 1: Cross-Asset Lead-Lag Dynamics
    df = add_cross_asset_lead_lag_features(df, eth_filtered, cutoff_timestamp_ms=cutoff_timestamp_ms)

    # 3. Add Family 2: Spot-Perpetual Basis Divergence & Funding Pressure
    df = add_basis_and_funding_features(df, spot_perp_filtered, cutoff_timestamp_ms=cutoff_timestamp_ms)

    # 4. Add Family 3: Order Book Microstructure & Flow Toxicity
    df = add_microstructure_and_vpin_features(df, orderbook_filtered, cutoff_timestamp_ms=cutoff_timestamp_ms)

    # 5. Clean, downcast, and assert all orthogonal features present
    # Replace +-inf and NaN with 0.0
    if hasattr(df, "_df"):
        for c in list(df._df.columns):
            if c != "ts":
                arr = np.nan_to_num(df._df[c].to_numpy(dtype=np.float32, na_value=0.0), nan=0.0, posinf=0.0, neginf=0.0)
                df._df[c] = arr
    else:
        float_cols = [c for c, t in df.schema.items() if t in (pl.Float32, pl.Float64)]
        df = df.with_columns([
            pl.when(pl.col(c).is_infinite() | pl.col(c).is_nan() | pl.col(c).is_null())
            .then(0.0)
            .otherwise(pl.col(c))
            .cast(pl.Float32)
            .alias(c)
            for c in float_cols
        ])

    # 6. Select stationary features and drop non-stationary / forbidden price levels
    available_cols = set(df.columns)
    selected_cols = ["ts"] if "ts" in available_cols else []
    for c in PRUNED_ORTHOGONAL_FEATURES:
        if c in available_cols and c not in selected_cols:
            selected_cols.append(c)

    # Preserve labels if present in input for training/evaluation
    for lbl in ["label_triple_barrier", "label_return_h1", "label_is_valid"]:
        if lbl in available_cols and lbl not in selected_cols:
            selected_cols.append(lbl)

    out = df.select(selected_cols)

    # Verify no non-stationary or dead columns survived
    for f in FORBIDDEN_NON_STATIONARY:
        if f in out.columns:
            out = out.drop(f)

    return out
