"""
Feature pruning per plan §5.4 (Phase 1): "run permutation importance and SHAP
analysis (reuse the GBM family from §7 for this) against the primary training
window, and drop any feature family that doesn't show a stable, monotone
importance signal across at least three non-overlapping time slices. Document
the final pruned feature list per model member."

Reuses models.gbm_family.train.train_classifier (the exact same LightGBM
direction classifier every other GBM-family fold already trains) on the
CLASSIFICATION_TARGET, evaluated across >=3 non-overlapping purged/embargoed
folds (common.cv.purged_walk_forward_splits — the same fold generator every
other training script in this repo uses, so "non-overlapping time slices"
here means the same thing it means everywhere else in this codebase).

Per fold this computes two independent importance signals:
  - permutation importance: shuffle one feature column at a time in the val
    set, re-predict, measure the drop in balanced accuracy vs. the
    unshuffled baseline (averaged over --n-repeats shuffles per feature).
  - SHAP importance: shap.TreeExplainer(booster) mean(|shap value|) per
    feature, averaged across classes, on a bounded subsample of the val set
    (SHAP on the full val set is not tractable at this row count).

Features are grouped into named "families" (FEATURE_FAMILIES below) rather
than judged individually — a single noisy column flipping sign fold-to-fold
is expected; what the plan asks to prune is a whole family that never earns
its place. A family's signal is deemed "stable" here if its permutation
importance is net-positive (shuffling it actually hurts accuracy) in >= 2 of
the >= 3 folds — the same "2 of 3 slices" bar the kill-criteria script
(validation/regime_holdout_eval.py) uses for member-level decisions, applied
here at feature-family granularity. This is a pragmatic reading of the plan's
"stable, monotone" wording (the plan does not give a numeric threshold); it
is stated explicitly here and in the output report rather than silently
assumed, per the plan's own §12.4 honest-gap-reporting philosophy.

This script only DOCUMENTS the pruned/kept family lists (checkpoints/
feature_pruning/metrics.json + pruned_features.json) — it deliberately does
not mutate common/feature_contract.py or any model's FEATURE_COLUMNS import,
since every existing train.py imports that contract directly and an
automatic prune here would silently change what every member trains on.
Wiring a pruned feature set into training is a follow-up config change once
a human reviews this report.

Usage:
    python -m validation.feature_pruning --n-folds 3 --val-days 7
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import polars as pl

import config
from common.cv import purged_walk_forward_splits
from common.data import filter_valid, load_days, to_xy
from common.feature_contract import CLASSIFICATION_TARGET, FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("validation.feature_pruning")

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints" / "feature_pruning"

# Ordered (prefix/exact-match -> family name); first match wins, "other" is
# the fallback for anything not covered. Derived directly from col_schema.json
# column names (common.feature_contract.FEATURE_COLUMNS), grouped by what
# part of the pipeline produces them (feature_engine/*.py) rather than by
# any formal spec, since the plan doesn't enumerate families explicitly.
FEATURE_FAMILIES: list[tuple[str, tuple[str, ...]]] = [
    ("calendar_time", (
        "hour_", "minute_", "day_of_week", "day_of_month", "month_", "quarter_",
        "week_of_year", "is_month_", "is_quarter_", "session_", "is_weekend",
        "friday_close_window", "minutes_to_funding", "minutes_from_funding",
        "pre_funding_1h", "post_funding_1h", "funding_cycle_phase",
    )),
    ("liquidation", ("liquidation_",)),
    ("htf_context", ("htf_",)),
    ("breakout_sr", ("breakout_", "resistance_distance_", "support_distance_")),
    ("distribution_stats", (
        "return_skew_", "return_kurtosis_", "return_autocorr_", "vol_clustering_",
        "hurst_proxy", "vol_regime_bucket",
    )),
    ("technical_indicators", (
        "rsi", "macd", "bollinger_", "atr", "plus_di", "minus_di", "adx",
    )),
    ("implied_vol_term", (
        "implied_vol", "iv_realized_vol_spread", "calendar_basis_", "term_structure_slope",
    )),
    ("cross_exchange", ("bybit_", "cross_exch_")),
    ("spot_cross_asset", ("spot_",)),
    ("basis_premium", ("index_price", "basis_pct", "mark_price", "premium_index")),
    ("funding_positioning", (
        "funding_rate", "open_interest", "toptrader_", "all_account_ls_ratio",
        "taker_ls_vol_ratio", "oi_roc_", "oi_change_pct", "volume_to_oi_ratio",
        "funding_oi_crowding_",
    )),
    ("orderflow_microstructure", (
        "order_flow_imbalance_", "ofi_", "proxy_spread_pct", "roll_spread_",
        "kyle_lambda_", "amihud_illiquidity_", "vpin_proxy_", "quote_resilience",
    )),
    ("volume", (
        "volume", "trade_count", "buy_volume", "sell_volume", "notional_volume",
        "intra_bar_price_std", "max_trade_notional", "trade_size_std", "volume_delta",
        "taker_buy_", "cvd", "large_trade_dominance", "trade_size_cv",
    )),
    ("price_action", (
        "open", "high", "low", "close", "log_return_", "momentum_", "realized_vol_",
        "price_pct_of_range_", "dist_from_high_", "dist_from_low_", "bar_range_pct",
        "bar_body_pct", "vwap",
    )),
]


def assign_family(col: str) -> str:
    for family, prefixes in FEATURE_FAMILIES:
        if col.startswith(prefixes):
            return family
    return "other"


FEATURE_TO_FAMILY: dict[str, str] = {c: assign_family(c) for c in FEATURE_COLUMNS}


def _shap_importance(booster, X: np.ndarray, feature_cols: list[str], max_rows: int = 2000) -> dict[str, float]:
    """Mean(|SHAP value|) per feature, averaged over classes, on a bounded
    subsample (TreeExplainer over the full val set is not tractable here)."""
    try:
        import shap
    except ImportError:
        log.warning("shap not installed — skipping SHAP importance for this fold (permutation importance still runs)")
        return {}

    rng = np.random.default_rng(0)
    if X.shape[0] > max_rows:
        idx = rng.choice(X.shape[0], size=max_rows, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    explainer = shap.TreeExplainer(booster)
    raw = explainer.shap_values(X_sample)

    if isinstance(raw, list):
        # older shap API: list of (n_samples, n_features) arrays, one per class
        stacked = np.stack([np.abs(a) for a in raw], axis=0)  # (n_classes, n_samples, n_features)
        mean_abs = stacked.mean(axis=(0, 1))
    else:
        arr = np.asarray(raw)
        if arr.ndim == 3:
            # newer shap API: (n_samples, n_features, n_classes)
            mean_abs = np.abs(arr).mean(axis=(0, 2))
        else:
            mean_abs = np.abs(arr).mean(axis=0)

    return {c: float(v) for c, v in zip(feature_cols, mean_abs)}


def _permutation_importance(booster, X: np.ndarray, y: np.ndarray, feature_cols: list[str], n_repeats: int) -> dict[str, float]:
    """Per-feature drop in balanced accuracy when that column is shuffled,
    averaged over n_repeats independent shuffles. Positive == the feature
    matters (shuffling it hurts); ~0 or negative == it doesn't."""
    from sklearn.metrics import balanced_accuracy_score

    rng = np.random.default_rng(0)
    baseline_pred = booster.predict(X, num_iteration=booster.best_iteration).argmax(axis=1)
    baseline_acc = balanced_accuracy_score(y, baseline_pred)

    importances: dict[str, float] = {}
    X_shuffled = X.copy()
    for j, col in enumerate(feature_cols):
        drops = []
        original = X_shuffled[:, j].copy()
        for _ in range(n_repeats):
            rng.shuffle(X_shuffled[:, j])
            pred = booster.predict(X_shuffled, num_iteration=booster.best_iteration).argmax(axis=1)
            acc = balanced_accuracy_score(y, pred)
            drops.append(baseline_acc - acc)
        X_shuffled[:, j] = original
        importances[col] = float(np.mean(drops))
    return importances


def run(n_folds: int, val_days: int, max_days: int | None, n_repeats: int, upload: bool) -> dict:
    if n_folds < 3:
        log.warning("n_folds=%d < 3 — plan §5.4 requires >= 3 non-overlapping time slices; forcing n_folds=3", n_folds)
        n_folds = 3

    from models.gbm_family.train import train_classifier

    df = filter_valid(load_days(max_days=max_days))
    val_size = val_days * config.BARS_PER_DAY
    splits = purged_walk_forward_splits(df, n_folds=n_folds, val_size=val_size)

    per_fold_permutation: list[dict[str, float]] = []
    per_fold_shap: list[dict[str, float]] = []
    fold_reports = []

    for split in splits:
        booster, metrics = train_classifier(split.train, split.val, num_class=3, target=CLASSIFICATION_TARGET)
        X_val, y_val, _ = to_xy(split.val, target_col=CLASSIFICATION_TARGET)

        perm = _permutation_importance(booster, X_val, y_val, FEATURE_COLUMNS, n_repeats)
        shap_imp = _shap_importance(booster, X_val, FEATURE_COLUMNS)

        per_fold_permutation.append(perm)
        per_fold_shap.append(shap_imp)
        fold_reports.append({
            "fold_index": split.fold_index,
            "direction_accuracy": metrics["accuracy"],
            "direction_balanced_accuracy": metrics["balanced_accuracy"],
        })
        log.info("Fold %d: direction_balanced_accuracy=%.4f (perm/shap importance computed over %d features)",
                  split.fold_index, metrics["balanced_accuracy"], len(FEATURE_COLUMNS))

    # Aggregate per-feature permutation importance -> per-family, per fold.
    families = sorted({FEATURE_TO_FAMILY[c] for c in FEATURE_COLUMNS})
    family_fold_perm: dict[str, list[float]] = {fam: [] for fam in families}
    family_fold_shap: dict[str, list[float]] = {fam: [] for fam in families}
    for fold_perm, fold_shap in zip(per_fold_permutation, per_fold_shap):
        fam_sums_perm: dict[str, list[float]] = {fam: [] for fam in families}
        fam_sums_shap: dict[str, list[float]] = {fam: [] for fam in families}
        for col in FEATURE_COLUMNS:
            fam = FEATURE_TO_FAMILY[col]
            fam_sums_perm[fam].append(fold_perm.get(col, 0.0))
            if fold_shap:
                fam_sums_shap[fam].append(fold_shap.get(col, 0.0))
        for fam in families:
            family_fold_perm[fam].append(float(np.mean(fam_sums_perm[fam])) if fam_sums_perm[fam] else 0.0)
            family_fold_shap[fam].append(float(np.mean(fam_sums_shap[fam])) if fam_sums_shap[fam] else 0.0)

    family_report = {}
    kept_families = []
    dropped_families = []
    for fam in families:
        perm_by_fold = family_fold_perm[fam]
        positive_folds = sum(1 for v in perm_by_fold if v > 0.0)
        stable = positive_folds >= 2  # ">= 2 of >= 3" bar, same as the member-level kill criteria
        family_report[fam] = {
            "n_features": sum(1 for c in FEATURE_COLUMNS if FEATURE_TO_FAMILY[c] == fam),
            "permutation_importance_by_fold": perm_by_fold,
            "shap_importance_by_fold": family_fold_shap[fam],
            "positive_folds": positive_folds,
            "n_folds": len(perm_by_fold),
            "stable": stable,
        }
        (kept_families if stable else dropped_families).append(fam)

    kept_features = [c for c in FEATURE_COLUMNS if FEATURE_TO_FAMILY[c] in kept_families]
    dropped_features = [c for c in FEATURE_COLUMNS if FEATURE_TO_FAMILY[c] in dropped_families]

    report = {
        "method": "permutation importance (balanced-accuracy drop) + SHAP (TreeExplainer, GBM direction "
                  "classifier), aggregated per feature family; a family is kept iff its permutation "
                  "importance is net-positive in >= 2 of the evaluated non-overlapping folds",
        "n_folds": len(splits),
        "n_repeats_per_feature": n_repeats,
        "fold_summaries": fold_reports,
        "families": family_report,
        "kept_families": kept_families,
        "dropped_families": dropped_families,
        "n_features_total": len(FEATURE_COLUMNS),
        "n_features_kept": len(kept_features),
        "n_features_dropped": len(dropped_features),
    }

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    (CHECKPOINT_DIR / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (CHECKPOINT_DIR / "pruned_features.json").write_text(json.dumps({
        "kept_features": kept_features,
        "dropped_features": dropped_features,
        "kept_families": kept_families,
        "dropped_families": dropped_families,
    }, indent=2), encoding="utf-8")
    log.info("Wrote feature-pruning report to %s (%d/%d features kept across %d/%d families)",
              CHECKPOINT_DIR, len(kept_features), len(FEATURE_COLUMNS), len(kept_families), len(families))

    if upload:
        from common.hf_model_upload import upload_best_checkpoint

        repo_id = upload_best_checkpoint(
            "feature-pruning",
            [CHECKPOINT_DIR / "metrics.json", CHECKPOINT_DIR / "pruned_features.json"],
            metrics=report,
        )
        log.info("Uploaded feature-pruning report to %s", repo_id)

    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-folds", type=int, default=3, help="plan §5.4 requires >= 3 non-overlapping slices")
    parser.add_argument("--val-days", type=int, default=config.VAL_DAYS)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--n-repeats", type=int, default=3, help="shuffles per feature for permutation importance")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    run(args.n_folds, args.val_days, args.max_days, args.n_repeats, args.upload)


if __name__ == "__main__":
    main()
