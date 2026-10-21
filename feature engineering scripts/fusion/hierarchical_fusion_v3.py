"""
feature engineering scripts/fusion/hierarchical_fusion_v3.py
Production implementation of the 4-Tier Hierarchical Fusion Engine,
Cross-Horizon Alignment Interaction Vectors & Shannon Negentropy Helpers.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal
import numpy as np


@dataclass(frozen=True)
class HierarchicalDecisionOutput:
    ts_ms: int
    macro_probs: tuple[float, float, float]       # [down, flat, up] for Tier 4 (4h-1d)
    meso_probs: tuple[float, float, float]        # [down, flat, up] for Tier 3 (15m-2h)
    tactical_probs: tuple[float, float, float]    # [down, flat, up] for Tier 2 (1m-5m)
    micro_probs: tuple[float, float, float]       # [down, flat, up] for Tier 1 (1s-10s)
    fused_swing_probs: tuple[float, float, float] # Calibrated [down, flat, up] for trade horizon
    primary_direction: int                        # 0 = SHORT, 1 = FLAT, 2 = LONG
    cross_horizon_alignment: float                # [-1.0, +1.0] coherence score
    expected_gross_move_bps: float                # Predicted swing excursion in bps
    expected_net_edge_bps: float                  # Expected move minus Maker/Hybrid fee hurdle
    meta_act_probability: float                   # Calibrated P(Trade is Net Profitable) in [0, 1]
    micro_sniper_trigger_prob: float              # P(Immediate 1s/10s Maker Entry is Optimal Now)
    target_holding_seconds: int                   # Dynamic horizon (e.g. 900s to 7200s)
    take_profit_bps: float                        # Dynamic volatility-scaled TP barrier
    stop_loss_bps: float                          # Dynamic volatility-scaled SL barrier
    kelly_position_fraction: float                # Volatility-targeted fractional Kelly size [0, 1]
    execution_mode: str                           # "NONE", "PASSIVE_MAKER_SNIPER", "URGENT_HYBRID", "WAIT_FOR_MICRO_SNIPER"


def shannon_negentropy_norm(probs: np.ndarray) -> np.ndarray:
    """Computes normalized certainty C_t = 1 - H(p)/ln(3) in [0, 1] for 3-class probabilities."""
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-7, 1.0)
    p = p / np.sum(p, axis=-1, keepdims=True)
    entropy = -np.sum(p * np.log(p), axis=-1)
    return np.clip(1.0 - entropy / np.log(3.0), 0.0, 1.0)


def compute_cross_horizon_alignment_features(
    p_fast: np.ndarray,
    p_slow: np.ndarray,
) -> np.ndarray:
    """
    Computes the 4-dimensional cross-horizon interaction vector between a fast tier
    and a slow tier: [coherence, conflict, joint_conviction, flat_veto_pressure].
    Handles both 1-D (3,) inputs and 2-D (N, 3) batch matrices.
    """
    pf = np.asarray(p_fast, dtype=np.float64)
    ps = np.asarray(p_slow, dtype=np.float64)
    is_1d = (pf.ndim == 1)
    if is_1d:
        pf = pf.reshape(1, -1)
        ps = ps.reshape(1, -1)

    coherence = pf[:, 2] * ps[:, 2] + pf[:, 0] * ps[:, 0]
    conflict = pf[:, 2] * ps[:, 0] + pf[:, 0] * ps[:, 2]

    edge_f = pf[:, 2] - pf[:, 0]
    edge_s = ps[:, 2] - ps[:, 0]
    cert_f = shannon_negentropy_norm(pf)
    cert_s = shannon_negentropy_norm(ps)

    joint_conviction = np.sign(edge_f * edge_s) * np.sqrt(np.abs(edge_f * edge_s)) * cert_f * cert_s
    flat_veto_pressure = 1.0 - (1.0 - pf[:, 1]) * (1.0 - ps[:, 1])

    out = np.column_stack([
        coherence,
        conflict,
        joint_conviction,
        flat_veto_pressure,
    ]).astype(np.float32)

    return out[0] if is_1d else out


class HierarchicalMultiHorizonFusionEngine:
    """
    Fuses Tier 4 (Macro), Tier 3 (Meso), Tier 2 (Tactical), and Tier 1 (Micro)
    predictions into a single calibrated trade decision, decoupling Swing Trade
    Authorization (Tier 4 + Tier 3 + Tier 2) from Instantaneous Maker Order
    Triggering (Tier 1).
    """

    def __init__(
        self,
        maker_roundtrip_fee_bps: float = 4.0,
        hybrid_roundtrip_fee_bps: float = 7.5,
        min_net_edge_buffer_bps: float = 6.0,
        min_meta_act_prob: float = 0.58,
        min_sniper_trigger_prob: float = 0.55,
        max_leverage_fraction: float = 1.0,
    ) -> None:
        self.maker_fee_bps = maker_roundtrip_fee_bps
        self.hybrid_fee_bps = hybrid_roundtrip_fee_bps
        self.edge_buffer_bps = min_net_edge_buffer_bps
        self.min_meta_act_prob = min_meta_act_prob
        self.min_sniper_trigger_prob = min_sniper_trigger_prob
        self.max_leverage_fraction = max_leverage_fraction

    def evaluate_multi_horizon_tick(
        self,
        ts_ms: int,
        p_macro: np.ndarray,      # shape (3,) [down, flat, up]
        p_meso: np.ndarray,       # shape (3,) [down, flat, up]
        p_tactical: np.ndarray,   # shape (3,) [down, flat, up]
        p_micro: np.ndarray,      # shape (3,) [down, flat, up]
        atr_15m_bps: float,
        atr_1h_bps: float,
        atr_10s_bps: float,
        micro_safe_fill_prob: float = 0.65,
        regime_vol_multiplier: float = 1.0,
    ) -> HierarchicalDecisionOutput:
        # 1. Compute pairwise cross-horizon alignment features
        align_3_4 = compute_cross_horizon_alignment_features(p_meso, p_macro)
        align_2_3 = compute_cross_horizon_alignment_features(p_tactical, p_meso)
        align_1_2 = compute_cross_horizon_alignment_features(p_micro, p_tactical)

        d_macro = float(p_macro[2] - p_macro[0])
        d_meso = float(p_meso[2] - p_meso[0])
        d_tact = float(p_tactical[2] - p_tactical[0])
        d_micro = float(p_micro[2] - p_micro[0])

        # 2. Top-Down Gating: Macro (Tier 4) conditions Meso (Tier 3) and Tactical (Tier 2)
        macro_conflict = (d_macro * d_meso) < -0.05
        macro_gate = 0.45 if macro_conflict else min(1.25, 1.0 + 0.50 * float(align_3_4[0]))

        # Combine Tier 4 (25%), Tier 3 (45%), Tier 2 (30%) for primary Swing Direction
        raw_swing_probs = 0.25 * p_macro + 0.45 * (p_meso * macro_gate) + 0.30 * p_tactical
        raw_swing_probs = np.clip(raw_swing_probs, 1e-6, None)
        fused_swing = raw_swing_probs / np.sum(raw_swing_probs)

        p_dn, p_fl, p_up = float(fused_swing[0]), float(fused_swing[1]), float(fused_swing[2])
        swing_edge_signed = p_up - p_dn
        swing_conviction = abs(swing_edge_signed) * (1.0 - 0.5 * p_fl)

        if abs(swing_edge_signed) < 0.08 or p_fl > 0.48:
            primary_dir = 1  # FLAT
        else:
            primary_dir = 2 if swing_edge_signed > 0 else 0

        # 3. Overall Cross-Horizon Alignment Score in [-1, +1]
        sign_target = 1.0 if primary_dir == 2 else (-1.0 if primary_dir == 0 else 0.0)
        coherence = sign_target * (0.35 * d_macro + 0.40 * d_meso + 0.25 * d_tact)
        cross_alignment = float(np.clip(coherence * 1.8, -1.0, 1.0))

        # 4. Dynamic Horizon & Volatility-Scaled Barrier Calculation (Anchored to Meso 15m/1h ATR!)
        effective_swing_atr_bps = max(15.0, 0.60 * atr_15m_bps + 0.40 * (atr_1h_bps / 2.0))
        tp_bps = float(np.clip(1.85 * effective_swing_atr_bps * (0.85 + 0.30 * max(0.0, cross_alignment)), 24.0, 180.0))
        sl_bps = float(np.clip(1.15 * effective_swing_atr_bps, 14.0, 95.0))

        # Expected gross move scales with conviction * TP barrier vs SL risk
        meta_logit = -0.35 + 4.2 * swing_conviction + 1.1 * cross_alignment
        meta_act_prob = float(np.clip(1.0 / (1.0 + math.exp(-meta_logit)), 0.02, 0.98))

        expected_gross_bps = meta_act_prob * tp_bps - (1.0 - meta_act_prob) * sl_bps
        expected_net_bps = expected_gross_bps - self.maker_fee_bps

        # 5. Tier 1 (1s/10s) Sniper Trigger Evaluation
        micro_agrees_with_swing = (sign_target * d_micro) >= -0.02
        is_pullback_inflection = ((sign_target * d_tact) < 0.05) and ((sign_target * d_micro) > 0.12)
        sniper_score = (
            0.45 * micro_safe_fill_prob
            + 0.35 * (0.5 + 0.5 * sign_target * d_micro)
            + (0.20 if is_pullback_inflection else 0.05 * max(0.0, float(align_1_2[0])))
        )
        sniper_trigger_prob = float(np.clip(sniper_score, 0.01, 0.99)) if micro_agrees_with_swing else 0.20

        # 6. Dynamic Holding Horizon (15m = 900s up to 2h = 7200s based on Macro/Meso dominance)
        if abs(d_macro) > 0.25 and (sign_target * d_macro) > 0:
            target_hold_sec = 3600  # 1 hour hold when aligned with strong Macro trend
        elif abs(d_meso) > 0.20:
            target_hold_sec = 1800  # 30 minute hold for standard Meso swing
        else:
            target_hold_sec = 900   # 15 minute hold for Tactical impulse

        # 7. Volatility-Targeted Fractional Kelly Position Sizing
        b_payoff_ratio = max(0.5, (tp_bps - self.maker_fee_bps) / max(1.0, sl_bps + self.maker_fee_bps))
        q_loss = 1.0 - meta_act_prob
        raw_kelly = max(0.0, (meta_act_prob * b_payoff_ratio - q_loss) / b_payoff_ratio)
        half_kelly = 0.50 * raw_kelly * regime_vol_multiplier * max(0.25, 0.5 + 0.5 * cross_alignment)

        # Gate execution authorization
        is_swing_valid = (
            primary_dir in (0, 2)
            and meta_act_prob >= self.min_meta_act_prob
            and expected_net_bps >= self.edge_buffer_bps
        )

        if not is_swing_valid:
            exec_mode = "NONE"
            final_size = 0.0
        elif expected_gross_bps >= (self.hybrid_fee_bps + 2.5 * self.edge_buffer_bps) and sniper_trigger_prob >= 0.80:
            exec_mode = "URGENT_HYBRID"
            final_size = float(np.clip(half_kelly, 0.05, self.max_leverage_fraction))
        elif sniper_trigger_prob >= self.min_sniper_trigger_prob:
            exec_mode = "PASSIVE_MAKER_SNIPER"
            final_size = float(np.clip(half_kelly, 0.05, self.max_leverage_fraction))
        else:
            exec_mode = "WAIT_FOR_MICRO_SNIPER"
            final_size = 0.0

        return HierarchicalDecisionOutput(
            ts_ms=ts_ms,
            macro_probs=(float(p_macro[0]), float(p_macro[1]), float(p_macro[2])),
            meso_probs=(float(p_meso[0]), float(p_meso[1]), float(p_meso[2])),
            tactical_probs=(float(p_tactical[0]), float(p_tactical[1]), float(p_tactical[2])),
            micro_probs=(float(p_micro[0]), float(p_micro[1]), float(p_micro[2])),
            fused_swing_probs=(p_dn, p_fl, p_up),
            primary_direction=primary_dir,
            cross_horizon_alignment=round(cross_alignment, 4),
            expected_gross_move_bps=round(expected_gross_bps, 2),
            expected_net_edge_bps=round(expected_net_bps, 2),
            meta_act_probability=round(meta_act_prob, 4),
            micro_sniper_trigger_prob=round(sniper_trigger_prob, 4),
            target_holding_seconds=target_hold_sec,
            take_profit_bps=round(tp_bps, 2),
            stop_loss_bps=round(sl_bps, 2),
            kelly_position_fraction=round(final_size, 4),
            execution_mode=exec_mode,
        )
