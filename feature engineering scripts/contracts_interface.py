"""
Interface contracts and validation routines for BTCUSDT 10-Second Alpha & Execution Overhaul.
Defined in PROJECT.md § Interface Contracts.
"""
from __future__ import annotations

from dataclasses import dataclass
import sys
import types


@dataclass(frozen=True)
class EnsembleInferenceResult:
    member_predictions: dict[str, float]
    pairwise_correlations: dict[tuple[str, str], float]
    primary_direction: int
    raw_ensemble_score: float
    expected_move_bps: float

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
    continuous_probability: float
    is_authorized: bool
    expected_move_bps: float
    hurdle_passed: bool
    chop_gate_passed: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "continuous_probability", float(self.continuous_probability))
        object.__setattr__(self, "is_authorized", bool(self.is_authorized))
        object.__setattr__(self, "expected_move_bps", float(self.expected_move_bps))
        object.__setattr__(self, "hurdle_passed", bool(self.hurdle_passed))
        object.__setattr__(self, "chop_gate_passed", bool(self.chop_gate_passed))

    def validate(self) -> None:
        if not (0.0 <= self.continuous_probability <= 1.0):
            raise ValueError(f"continuous_probability {self.continuous_probability} out of [0, 1]")
        if self.expected_move_bps < 0.0:
            raise ValueError(f"expected_move_bps cannot be negative: {self.expected_move_bps}")
        if self.is_authorized and (not self.hurdle_passed or not self.chop_gate_passed):
            raise ValueError("is_authorized cannot be True if hurdle or chop gate failed")


# Auto-alias tests.contracts in sys.modules so any legacy import succeeds
if "tests" not in sys.modules:
    sys.modules["tests"] = types.ModuleType("tests")
if "tests.contracts" not in sys.modules:
    m = types.ModuleType("tests.contracts")
    m.MetaLabelerOutput = MetaLabelerOutput
    m.EnsembleInferenceResult = EnsembleInferenceResult
    sys.modules["tests.contracts"] = m
