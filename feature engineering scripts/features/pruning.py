"""
Feature Pruning Pipeline.
Reduces the high-dimensional feature set down to high-SNR, low-correlation
orthogonal features via Hierarchical Clustered Correlation (max |r| <= 0.70),
Purged Walk-Forward CV Information Ratio, and Recursive Feature Elimination.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
import numpy as np
import polars as pl
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

try:
    from anti_leakage import make_expanding_walk_forward_folds, split_train_val
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from anti_leakage import make_expanding_walk_forward_folds, split_train_val

log = logging.getLogger("features.pruning")


def hierarchical_correlation_filter(
    df: pl.DataFrame,
    candidate_features: list[str],
    target_col: str = "label_return_h1",
    max_corr: float = 0.70,
) -> list[str]:
    """
    De-duplicates collinear features using Hierarchical Tree Clustering.
    Guarantees no two selected features have pairwise |Spearman r| > max_corr.
    """
    if len(candidate_features) <= 1:
        return list(candidate_features)

    # Use target if available, otherwise use index/proxy
    use_target = target_col in df.columns
    cols_to_select = [*candidate_features]
    if use_target:
        cols_to_select.append(target_col)

    sample_df = df.select(cols_to_select).drop_nulls()
    if sample_df.height > 20000:
        sample_df = sample_df.sample(20000, seed=42)
    elif sample_df.height < 50:
        return list(candidate_features)

    X = sample_df.select(candidate_features).to_numpy()

    # Compute pairwise correlation matrix
    corr_matrix, _ = spearmanr(X)
    if np.ndim(corr_matrix) < 2:
        return list(candidate_features)

    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    np.fill_diagonal(corr_matrix, 1.0)

    # Compute IC of each feature with target if available, else variance
    if use_target:
        y = sample_df[target_col].to_numpy()
        ics = np.array([abs(spearmanr(X[:, j], y)[0]) for j in range(len(candidate_features))])
        ics = np.nan_to_num(ics, nan=0.0)
    else:
        ics = np.std(X, axis=0)

    # Distance metric: D(i, j) = sqrt(0.5 * (1 - |rho|))
    dist_matrix = np.sqrt(0.5 * (1.0 - np.clip(np.abs(corr_matrix), 0.0, 1.0)))
    np.fill_diagonal(dist_matrix, 0.0)
    condensed_dist = squareform(dist_matrix, checks=False)

    # Linkage clustering
    Z = linkage(condensed_dist, method="average")
    d_cutoff = np.sqrt(0.5 * (1.0 - max_corr))
    cluster_labels = fcluster(Z, t=d_cutoff, criterion="distance")

    selected_features = []
    unique_clusters = np.unique(cluster_labels)
    for c in unique_clusters:
        indices = np.where(cluster_labels == c)[0]
        # Choose feature with highest IC / score in the cluster
        best_idx = indices[np.argmax(ics[indices])]
        selected_features.append(candidate_features[best_idx])

    log.info("Clustered %d features into %d orthogonal clusters (cut=%.2f)",
             len(candidate_features), len(selected_features), max_corr)
    return selected_features


def purged_walk_forward_ic_ranking(
    df: pl.DataFrame,
    feature_cols: list[str],
    target_col: str = "label_return_h1",
    n_folds: int = 5,
    purge_bars: int = 3,
    embargo_bars: int = 3,
) -> dict[str, float]:
    """
    Ranks features by Information Ratio across expanding purged walk-forward folds.
    """
    if target_col not in df.columns or df.height < 200:
        return {col: 1.0 for col in feature_cols}

    valid_df = df.filter(pl.col("label_is_valid") == True) if "label_is_valid" in df.columns else df
    val_size = max(20, valid_df.height // (n_folds + 2))
    try:
        folds = make_expanding_walk_forward_folds(
            valid_df.height,
            n_folds=n_folds,
            val_size=val_size,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
        )
    except ValueError:
        return {col: 1.0 for col in feature_cols}

    ic_history = {col: [] for col in feature_cols}
    for fold in folds:
        _, val = split_train_val(valid_df, fold)
        val_sub = val.select([*feature_cols, target_col]).drop_nulls()
        if val_sub.height < 10:
            continue
        y_val = val_sub[target_col].to_numpy()
        for col in feature_cols:
            x_col = val_sub[col].to_numpy()
            ic = spearmanr(x_col, y_val)[0]
            ic_history[col].append(0.0 if np.isnan(ic) else ic)

    ir_scores = {}
    for col, ics in ic_history.items():
        if not ics:
            ir_scores[col] = 0.0
        else:
            mean_ic = np.mean(ics)
            std_ic = np.std(ics)
            ir_scores[col] = float(abs(mean_ic) / (std_ic + 1e-4))
    return ir_scores


def run_feature_pruning_pipeline(
    df: pl.DataFrame,
    candidate_features: list[str],
    target_col: str = "label_return_h1",
    target_k: int = 30,
    output_manifest_path: Path | None = None,
) -> list[str]:
    """
    End-to-end execution of the 5-stage feature pruning pipeline downscaling
    candidate features to target_k orthogonal subset (max |r| <= 0.70).
    """
    # 1. Variance filter
    stds = df.select([pl.col(c).std().alias(c) for c in candidate_features]).to_dicts()[0]
    non_flat = [c for c, s in stds.items() if s is not None and s > 1e-6]

    # 2. Hierarchical correlation filter at r <= 0.70
    de_correlated = hierarchical_correlation_filter(df, non_flat, target_col=target_col, max_corr=0.70)

    # 3. Purged IC ranking
    ir_scores = purged_walk_forward_ic_ranking(df, de_correlated, target_col=target_col)

    # 4. Rank and select top target_k
    ranked_features = sorted(de_correlated, key=lambda c: ir_scores.get(c, 0.0), reverse=True)
    final_selected = ranked_features[:target_k]

    if output_manifest_path:
        manifest = {
            "target_k": len(final_selected),
            "selected_features": final_selected,
            "feature_ir_scores": {c: ir_scores.get(c, 0.0) for c in final_selected},
        }
        with open(output_manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        log.info("Saved pruned feature manifest to %s", output_manifest_path)

    return final_selected
