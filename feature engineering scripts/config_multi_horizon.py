"""
feature engineering scripts/config_multi_horizon.py
Single source of truth for Project-IMPJR v3.0 across all 14 resolutions:
  - Tier 4 (Macro Structural): 4h, 6h, 8h, 12h, 1d
  - Tier 3 (Meso Swing):       15m, 30m, 1h, 2h
  - Tier 2 (Tactical Impulse): 1m, 3m, 5m
  - Tier 1 (Micro Sniper):     1s, 10s
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", str(Path(__file__).resolve().parent.parent)))
DATA_VISION_ROOT = Path(os.environ.get("BINANCE_DATA_ROOT", PROJECT_ROOT / "data"))
MULTI_TF_OUTPUT_ROOT = Path(
    os.environ.get("MULTI_TF_OUTPUT_ROOT", PROJECT_ROOT / "feature engineering scripts" / "output_multi_tf")
)

# Primary and Cross-Asset Traded Universe
PRIMARY_SYMBOL_UM = "BTCUSDT"
PRIMARY_SYMBOL_CM = "BTCUSD_PERP"
PRIMARY_SYMBOL_SPOT = "BTCUSDT"
BVOL_SYMBOL = "BTCBVOLUSDT"
CROSS_ASSET_UNIVERSE = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]

# All 14 Supported Resolutions and Exact Durations in Seconds
TIMEFRAME_SECONDS: dict[str, int] = {
    "1s": 1,
    "10s": 10,
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
}

TIER_TIMEFRAMES: dict[str, list[str]] = {
    "tier4_macro": ["4h", "6h", "8h", "12h", "1d"],
    "tier3_meso": ["15m", "30m", "1h", "2h"],
    "tier2_tactical": ["1m", "3m", "5m"],
    "tier1_micro": ["1s", "10s"],
}


@dataclass(frozen=True)
class TierPipelineSpec:
    tier_name: str
    base_grid_interval: str
    constituent_intervals: tuple[str, ...]
    ffd_order_d: float
    lookback_bars: int
    forecast_horizon_bars: int
    triple_barrier_pt_mult: float
    triple_barrier_sl_mult: float
    min_gross_pt_bps: float
    min_gross_sl_bps: float
    purge_seconds: int
    embargo_seconds: int


TIER_SPECS: dict[str, TierPipelineSpec] = {
    "tier4_macro": TierPipelineSpec(
        tier_name="tier4_macro",
        base_grid_interval="4h",
        constituent_intervals=("4h", "6h", "8h", "12h", "1d"),
        ffd_order_d=0.28,
        lookback_bars=240,             # 240 x 4h = 40 days lookback per sequence window
        forecast_horizon_bars=6,       # 6 x 4h = 24 hours macro horizon
        triple_barrier_pt_mult=2.0,
        triple_barrier_sl_mult=1.2,
        min_gross_pt_bps=65.0,         # Strictly >= 65.0 bps gross move required
        min_gross_sl_bps=40.0,
        purge_seconds=48 * 3600,       # 48-hour strict purge window
        embargo_seconds=24 * 3600,     # 24-hour strict embargo window
    ),
    "tier3_meso": TierPipelineSpec(
        tier_name="tier3_meso",
        base_grid_interval="15m",
        constituent_intervals=("15m", "30m", "1h", "2h"),
        ffd_order_d=0.35,
        lookback_bars=192,             # 192 x 15m = 48 hours lookback
        forecast_horizon_bars=8,       # 8 x 15m = 2 hours primary swing horizon
        triple_barrier_pt_mult=1.8,
        triple_barrier_sl_mult=1.1,
        min_gross_pt_bps=28.0,         # Strictly >= 28.0 bps (7x larger than 4.0 bps Maker fee!)
        min_gross_sl_bps=16.0,
        purge_seconds=4 * 3600,        # 4-hour strict purge window
        embargo_seconds=4 * 3600,      # 4-hour strict embargo window
    ),
    "tier2_tactical": TierPipelineSpec(
        tier_name="tier2_tactical",
        base_grid_interval="1m",
        constituent_intervals=("1m", "3m", "5m"),
        ffd_order_d=0.42,
        lookback_bars=720,             # 720 x 1m = 12 hours lookback
        forecast_horizon_bars=15,      # 15 x 1m = 15 minutes tactical horizon
        triple_barrier_pt_mult=1.8,
        triple_barrier_sl_mult=1.2,
        min_gross_pt_bps=14.0,         # Strictly >= 14.0 bps tactical impulse floor
        min_gross_sl_bps=9.0,
        purge_seconds=30 * 60,         # 30-minute strict purge window
        embargo_seconds=30 * 60,       # 30-minute strict embargo window
    ),
    "tier1_micro": TierPipelineSpec(
        tier_name="tier1_micro",
        base_grid_interval="10s",
        constituent_intervals=("1s", "10s"),
        ffd_order_d=0.50,
        lookback_bars=180,             # 180 x 10s = 30 minutes microstructure context
        forecast_horizon_bars=3,       # 3 x 10s = 30 seconds sniper entry/exit window
        triple_barrier_pt_mult=1.6,
        triple_barrier_sl_mult=1.2,
        min_gross_pt_bps=6.0,
        min_gross_sl_bps=4.0,
        purge_seconds=60,              # 60-second purge window
        embargo_seconds=60,            # 60-second embargo window
    ),
}

# Execution & Fee Constants (Binance USD(S)-M Futures VIP 0)
MAKER_FEE_BPS_LEG = 2.0                # 0.02% Maker (1.8 bps with 10% BNB discount)
TAKER_FEE_BPS_LEG = 5.0                # 0.05% Taker (4.5 bps with 10% BNB discount)
MAKER_ROUNDTRIP_FEE_BPS = 4.0
HYBRID_ROUNDTRIP_FEE_BPS = 7.0
TAKER_ROUNDTRIP_FEE_BPS = 10.0

# Universal Anti-Leakage Freeze Window Cutoff
EXECUTION_FREEZE_OFFSET_MS = 1000      # t_event <= T_close - 1000ms
