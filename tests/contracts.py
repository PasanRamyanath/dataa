"""
Interface contracts and validation routines for BTCUSDT 10-Second Alpha & Execution Overhaul.
Defined in PROJECT.md § Interface Contracts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable
import polars as pl
import numpy as np


@dataclass(frozen=True)
class EnsembleInferenceResult:
    member_predictions: dict[str, float]  # {'microstructure': p1, 'cross_asset': p2, 'basis': p3}
    pairwise_correlations: dict[tuple[str, str], float]  # All values <= 0.70
    primary_direction: int  # 0: SHORT, 1: FLAT, 2: LONG
    raw_ensemble_score: float  # [-1.0, 1.0]
    expected_move_bps: float  # predicted absolute excursion

    def validate(self) -> None:
        if self.primary_direction not in (0, 1, 2):
            raise ValueError(f"Invalid primary_direction: {self.primary_direction}, expected 0, 1, or 2")
        if not (-1.0001 <= self.raw_ensemble_score <= 1.0001):
            raise ValueError(f"raw_ensemble_score {self.raw_ensemble_score} out of [-1.0, 1.0]")
        if self.expected_move_bps < 0.0:
            raise ValueError(f"expected_move_bps cannot be negative: {self.expected_move_bps}")
        for pair, corr in self.pairwise_correlations.items():
            if not (-1.0001 <= corr <= 1.0001):
                raise ValueError(f"Correlation for pair {pair} out of range: {corr}")
            if corr > 0.70001:
                raise ValueError(f"Pairwise correlation {corr:.4f} for pair {pair} exceeds 0.70 threshold")


@dataclass(frozen=True)
class MetaLabelerOutput:
    continuous_probability: float  # [0.0, 1.0] smoothly calibrated
    is_authorized: bool  # True if continuous_prob >= threshold AND hurdle passed
    expected_move_bps: float
    hurdle_passed: bool  # expected_move >= 6.0 bps
    chop_gate_passed: bool  # sigma_15m >= 15.0 bps and mean_10s_move >= 1.0 bps

    def __post_init__(self) -> None:
        object.__setattr__(self, "continuous_probability", float(self.continuous_probability))
        object.__setattr__(self, "is_authorized", bool(self.is_authorized))
        object.__setattr__(self, "expected_move_bps", float(self.expected_move_bps))
        object.__setattr__(self, "hurdle_passed", bool(self.hurdle_passed))
        object.__setattr__(self, "chop_gate_passed", bool(self.chop_gate_passed))

    def validate(self) -> None:
        if not (0.0 <= self.continuous_probability <= 1.0):
            raise ValueError(f"continuous_probability {self.continuous_probability} out of [0, 1]")
        if self.is_authorized and not (self.hurdle_passed and self.chop_gate_passed):
            raise ValueError("Signal cannot be authorized if hurdle or chop gate fails")
        if self.hurdle_passed and self.expected_move_bps < 6.0 - 1e-6:
            raise ValueError(f"hurdle_passed is True but expected_move_bps={self.expected_move_bps} < 6.0")


@dataclass(frozen=True)
class MakerOrderIntent:
    symbol: str  # "BTCUSDT"
    side: str  # "BUY" or "SELL"
    order_type: str  # "LIMIT"
    time_in_force: str  # "GTX" (Post-Only)
    limit_price: float  # BBO Best Bid for BUY, Best Ask for SELL
    quantity: float
    max_holding_bars: int  # 3 bars (30 seconds)
    take_profit_bps: float  # k_tp * ATR_10s
    stop_loss_bps: float    # k_sl * ATR_10s
    queue_ahead_qty: float  # Quantity resting ahead in book

    def validate(self) -> None:
        if self.symbol != "BTCUSDT":
            raise ValueError(f"Invalid symbol: {self.symbol}")
        if self.side not in ("BUY", "SELL"):
            raise ValueError(f"Invalid side: {self.side}")
        if self.order_type != "LIMIT":
            raise ValueError(f"Invalid order_type: {self.order_type}, must be LIMIT")
        if self.time_in_force != "GTX":
            raise ValueError(f"Invalid time_in_force: {self.time_in_force}, must be GTX (Post-Only)")
        if self.limit_price <= 0:
            raise ValueError(f"Invalid limit_price: {self.limit_price}")
        if self.max_holding_bars not in (3, 6, 12, 18):
            raise ValueError(f"max_holding_bars must be 3, 6, 12, or 18 bars (30-180 seconds), got {self.max_holding_bars}")
        if self.take_profit_bps <= 0 or self.stop_loss_bps <= 0:
            raise ValueError("Barriers must be strictly positive")
        if self.queue_ahead_qty < 0:
            raise ValueError("queue_ahead_qty cannot be negative")


@dataclass(frozen=True)
class ExecutionSimulationEvent:
    bar_index: int
    timestamp_ms: int
    signal_type: str  # "LONG" or "SHORT"
    entry_price: float  # Fill price at BBO
    exit_price: float   # Fill price at barrier or horizon expiry
    duration_seconds: float  # <= 180.0s
    gross_return_bps: float
    maker_fee_bps: float  # Exactly 4.0 bps RT (2.0 bps entry + 2.0 bps exit)
    net_return_bps: float  # gross_return_bps - 4.0 bps
    exit_reason: str  # "TAKE_PROFIT", "STOP_LOSS", "TIME_EXPIRY", "ADVERSE_CANCEL"
    is_maker: bool  # Must be True for all valid trades

    def validate(self) -> None:
        if self.signal_type not in ("LONG", "SHORT"):
            raise ValueError(f"Invalid signal_type: {self.signal_type}")
        if self.duration_seconds > 180.0 + 1e-4:
            raise ValueError(f"Duration {self.duration_seconds} exceeds 180.0s max structural horizon")
        if not np.isclose(self.maker_fee_bps, 4.0, atol=1e-5):
            raise ValueError(f"Maker fee must be 4.0 bps RT, got {self.maker_fee_bps}")
        expected_net = self.gross_return_bps - 4.0
        if not np.isclose(self.net_return_bps, expected_net, atol=1e-4):
            raise ValueError(f"net_return_bps {self.net_return_bps} != gross {self.gross_return_bps} - 4.0")
        if self.exit_reason not in ("TAKE_PROFIT", "STOP_LOSS", "TIME_EXPIRY", "ADVERSE_CANCEL"):
            raise ValueError(f"Invalid exit_reason: {self.exit_reason}")
        if not self.is_maker:
            raise ValueError("is_maker must be True for all maker executions")


@runtime_checkable
class OrthogonalFeatureComputer(Protocol):
    def __call__(
        self,
        btc_bars_df: pl.DataFrame,
        eth_bars_df: pl.DataFrame,
        spot_perp_df: pl.DataFrame,
        orderbook_df: pl.DataFrame,
        cutoff_timestamp_ms: int,
    ) -> pl.DataFrame:
        """Computes pruned orthogonal feature set strictly adhering to t_event <= cutoff <= T_close - 1000ms."""
        ...
