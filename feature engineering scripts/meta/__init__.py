"""
Meta-labeling package for BTCUSDT 10-Second Alpha & Execution Overhaul.
"""
from .continuous_meta_labeler import (
    ContinuousMetaLabeler,
    calculate_ece,
    calculate_brier_score,
    calibrate_probabilities,
)

__all__ = [
    "ContinuousMetaLabeler",
    "calculate_ece",
    "calculate_brier_score",
    "calibrate_probabilities",
]
