"""
feature engineering scripts/models_v3/tier1_micro/survival_queue_hazard.py
Tier 1 Microstructure Queue-Depletion & Non-Adverse Passive Fill Survival Hazard Model (new_plan.md §11.3, §19.3).
"""
from __future__ import annotations

import math
import numpy as np


class MicroQueueSurvivalHazardModel:
    """
    Estimates the probability P(T_fill <= tau_horizon AND Non-Adverse Post-Fill Drift)
    for a passive Post-Only (GTX) limit order resting at the Best Bid / Best Offer (BBO).
    Combines FIFO queue depletion hazard with VPIN / OFI adverse selection filtering.
    """

    def __init__(self, base_hazard_rate: float = 0.14, adverse_toxicity_penalty: float = 0.65) -> None:
        self.base_hazard_rate = base_hazard_rate
        self.adverse_toxicity_penalty = adverse_toxicity_penalty

    def predict_safe_fill_probability(
        self,
        queue_ahead_btc: float,
        avg_10s_trade_vol_btc: float,
        ofi_l1_norm: float,
        vpin_toxicity: float,
        spread_ticks: float = 1.0,
        horizon_seconds: float = 25.0,
    ) -> float:
        """
        Computes P(Safe Passive Fill within `horizon_seconds`) in [0.01, 0.99].
        """
        q_ahead = max(0.05, float(queue_ahead_btc))
        vol_rate_per_sec = max(0.02, float(avg_10s_trade_vol_btc) / 10.0)

        # Expected queue clearance time under Poisson / Gamma trade arrival
        expected_clear_sec = q_ahead / max(vol_rate_per_sec * 0.55, 1e-4)
        lam = (float(horizon_seconds) / max(expected_clear_sec, 0.5)) * (1.0 + 0.25 * float(ofi_l1_norm))
        p_raw_fill = 1.0 - math.exp(-max(0.05, lam))

        # Probability that the fill is NOT toxic (non-adverse selection)
        vpin_clamped = min(max(float(vpin_toxicity), 0.0), 0.99)
        p_non_adverse = math.exp(-self.adverse_toxicity_penalty * max(0.0, vpin_clamped - 0.35))

        spread_bonus = min(1.10, 1.0 + 0.03 * max(0.0, float(spread_ticks) - 1.0))
        safe_prob = p_raw_fill * p_non_adverse * spread_bonus
        return float(np.clip(safe_prob, 0.05, 0.98))
