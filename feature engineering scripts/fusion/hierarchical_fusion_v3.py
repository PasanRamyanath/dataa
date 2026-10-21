"""
feature engineering scripts/fusion/hierarchical_fusion_v3.py
Cross-Horizon Alignment Interaction Vector & Shannon Negentropy Helpers
used by SupervisedMultiHorizonOOFStacker and HierarchicalSniperExecutionEngine.
"""
from __future__ import annotations

import numpy as np


def shannon_negentropy_norm(probs: np.ndarray) -> np.ndarray:
    """Computes normalized certainty C_t = 1 - H(p)/ln(3) in [0, 1]."""
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-7, 1.0)
    p /= p.sum(axis=-1, keepdims=True)
    entropy = -np.sum(p * np.log(p), axis=-1)
    return np.clip(1.0 - entropy / np.log(3.0), 0.0, 1.0)


def compute_cross_horizon_alignment_features(
    p_fast: np.ndarray,
    p_slow: np.ndarray,
) -> np.ndarray:
    """
    Computes the 4-dimensional cross-horizon interaction vector between a fast tier
    and a slow tier: [coherence, conflict, joint_conviction, flat_veto_pressure].
    """
    pf = np.asarray(p_fast, dtype=np.float64)
    ps = np.asarray(p_slow, dtype=np.float64)

    coherence = pf[:, 2] * ps[:, 2] + pf[:, 0] * ps[:, 0]
    conflict = pf[:, 2] * ps[:, 0] + pf[:, 0] * ps[:, 2]

    edge_f = pf[:, 2] - pf[:, 0]
    edge_s = ps[:, 2] - ps[:, 0]
    cert_f = shannon_negentropy_norm(pf)
    cert_s = shannon_negentropy_norm(ps)

    joint_conviction = np.sign(edge_f * edge_s) * np.sqrt(np.abs(edge_f * edge_s)) * cert_f * cert_s
    flat_veto_pressure = 1.0 - (1.0 - pf[:, 1]) * (1.0 - ps[:, 1])

    return np.column_stack([
        coherence,
        conflict,
        joint_conviction,
        flat_veto_pressure,
    ]).astype(np.float32)
