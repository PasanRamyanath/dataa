"""
Modular Feature Engineering Package (Milestone 1).
Exposes orthogonal feature generators and unified pipeline:
- compute_orthogonal_features
- add_cross_asset_lead_lag_features
- add_basis_and_funding_features
- add_microstructure_and_vpin_features
- hierarchical_correlation_filter
- run_feature_pruning_pipeline
"""
from __future__ import annotations

from .cross_asset import add_cross_asset_lead_lag_features
from .basis_funding import add_basis_and_funding_features
from .microstructure_vpin import add_microstructure_and_vpin_features
from .pruning import (
    hierarchical_correlation_filter,
    purged_walk_forward_ic_ranking,
    run_feature_pruning_pipeline,
)
from .pipeline import compute_orthogonal_features, PRUNED_ORTHOGONAL_FEATURES

__all__ = [
    "compute_orthogonal_features",
    "add_cross_asset_lead_lag_features",
    "add_basis_and_funding_features",
    "add_microstructure_and_vpin_features",
    "hierarchical_correlation_filter",
    "purged_walk_forward_ic_ranking",
    "run_feature_pruning_pipeline",
    "PRUNED_ORTHOGONAL_FEATURES",
]
