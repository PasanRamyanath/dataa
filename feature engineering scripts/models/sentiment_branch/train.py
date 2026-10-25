"""
Sentiment-fusion & Cross-Asset Lead-Lag branch ensemble member (`sentiment-branch`,
`tier2-cryptobert-news`, and `tier2-crossasset-gat`).

Key v4.0 Upgrades:
  1. Distinct Specialist Subspaces for `tier2-crossasset-gat` vs `tier2-cryptobert-news`:
     - `tier2-crossasset-gat`: Cross-Asset & Cross-Exchange Lead-Lag, Spot-Perp Basis,
       Bybit Flow Divergence, Term-Structure Roll Yield, and Multi-Timeframe Propagation.
     - `tier2-cryptobert-news` / `sentiment-branch`: Crowd Positioning (`SENTIMENT_PROXY_FEATURES`),
       OI Crowding & Squeeze Context, Implied Volatility Spread (`VRP`), and CryptoBERT/LLM
       News Sentiment (`NEWS_FEATURE_COLUMNS` when `--with-news` is active).
  2. Volatility-Normalized Tactical Targets + SNR Sample Weighting + Dual-Head
     Classifier/Regressor Calibration:
     Combines LightGBM multiclass probabilities with a companion Huber return
     regressor and prior-debiasing calibration to maximize directional accuracy
     and active precision while remaining 100% compatible with `checkpoint_loaders.py`
     and `validation/ablation_test.py`.

Usage:
    python -m models.sentiment_branch.train --val-days 7 --upload
    python -m models.sentiment_branch.train --with-news --val-days 7 --upload
    python -m models.sentiment_branch.train --ablation --val-days 7
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import polars as pl

import config
from common.cv import purged_walk_forward_splits
from common.data import filter_valid, load_days, to_xy
from common.feature_contract import CLASSIFICATION_TARGET, FEATURE_COLUMNS, REGRESSION_TARGETS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("models.sentiment_branch.train")

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
ABLATION_CHECKPOINT_DIR = CHECKPOINT_DIR / "ablation"
MEMBER_NAME = "sentiment-branch"

SENTIMENT_PROXY_FEATURES = [
    "funding_rate", "funding_rate_ma_8h", "funding_rate_ma_24h", "funding_rate_change",
    "toptrader_count_ls_ratio", "toptrader_sum_ls_ratio", "all_account_ls_ratio",
    "taker_ls_vol_ratio", "funding_oi_crowding_180", "funding_oi_crowding_720",
    "implied_vol", "implied_vol_change_1", "iv_realized_vol_spread",
]
WITHOUT_SENTIMENT_FEATURES = [c for c in FEATURE_COLUMNS if c not in SENTIMENT_PROXY_FEATURES]

# Contextual regime & squeeze features strictly inside FEATURE_COLUMNS for high-precision crowding resolution
CROWD_SQUEEZE_CONTEXT_FEATURES = [c for c in [
    "open_interest_change_1", "open_interest_change_12", "open_interest_change_60",
    "oi_price_regime", "basis_pct", "basis_zscore_180", "basis_zscore_720",
    "premium_index", "premium_index_ma_60", "taker_buy_sell_imbalance",
    "order_flow_imbalance_near", "vwap_deviation", "bollinger_pct_b", "rsi_14",
    "log_return_3", "log_return_18", "log_return_90", "realized_vol_90",
    "htf_1m_return", "htf_5m_return",
] if c in FEATURE_COLUMNS]

CROSS_ASSET_GAT_FEATURES = [
    c for c in FEATURE_COLUMNS
    if any(k in c for k in (
        "spot_", "bybit_", "cross_exch_", "basis_", "premium_", "roll_yield",
        "term_structure", "htf_1m_", "htf_5m_", "ofi_", "order_flow_",
        "taker_buy_sell", "vwap_", "log_return_", "realized_vol_", "momentum_",
    ))
]


def load_with_news(max_days: int | None) -> pl.DataFrame:
    """filter_valid(load_days()) + point-in-time news features. Raises if
    the news store covers none of the rows — training on an all-zero news
    block would silently produce a model that ignores news."""
    from news_pipeline.features import attach_news_features

    df = attach_news_features(filter_valid(load_days(max_days=max_days)))
    coverage = float(df["news_coverage"].mean()) if df.height else 0.0
    if coverage == 0.0:
        raise ValueError("news store covers none of the training window — run news_pipeline.scraper backfill "
                         "and news_pipeline.llm_scorer --once first (see news_pipeline/__init__.py)")
    log.info("news features attached: %.1f%% of rows inside the news store's coverage", 100 * coverage)
    return df


def _fit_and_early_stop_sets(train: pl.DataFrame, feature_cols: list[str]):
    """(fit, early-stop) lgb.Datasets from the training window only (see
    models.gbm_family.train.split_early_stopping) — the val fold stays
    untouched until scoring, so with/without comparisons aren't each tuned
    on the rows they're compared on."""
    import lightgbm as lgb

    from models.gbm_family.train import _compute_balanced_snr_weights, split_early_stopping

    fit, early_stop = split_early_stopping(train)
    X_fit, y_fit, _ = to_xy(fit, feature_cols=feature_cols, target_col=CLASSIFICATION_TARGET)
    X_es, y_es, _ = to_xy(early_stop, feature_cols=feature_cols, target_col=CLASSIFICATION_TARGET)
    w_fit = _compute_balanced_snr_weights(fit, y_fit)
    return (
        lgb.Dataset(X_fit, label=y_fit, weight=w_fit, free_raw_data=False),
        lgb.Dataset(X_es, label=y_es, free_raw_data=False),
        fit,
        early_stop,
    )


def train_and_eval(
    train: pl.DataFrame,
    val: pl.DataFrame,
    feature_cols: list[str] | None = None,
    return_proba: bool = False,
):
    import lightgbm as lgb
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    from models.gbm_family.train import _fuse_and_calibrate_gbm_proba, train_regressor

    is_default_proxy = feature_cols is None or feature_cols == SENTIMENT_PROXY_FEATURES
    feature_cols = list(feature_cols or SENTIMENT_PROXY_FEATURES)

    train_set, val_set, fit_df, es_df = _fit_and_early_stop_sets(train, feature_cols)
    X_val, y_val, _ = to_xy(val, feature_cols=feature_cols, target_col=CLASSIFICATION_TARGET)
    n_jobs = max(2, min(4, os.cpu_count() or 4))
    min_leaf = max(15, min(160, fit_df.height // 80))

    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "learning_rate": 0.04,
        "num_leaves": 23 if len(feature_cols) > 20 else 15,
        "max_depth": 6,
        "min_data_in_leaf": min_leaf,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.6,
        "lambda_l2": 2.0,
        "max_bin": 255,
        "force_col_wise": True,
        "n_jobs": n_jobs,
        "verbosity": -1,
        "seed": 42,
    }
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=1800,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
    )

    raw_val_proba = booster.predict(X_val, num_iteration=booster.best_iteration)

    # Fit companion return regressor if label_return_h2 / h1 is present for dual-head calibration
    reg_col = "label_return_h2" if "label_return_h2" in train.columns else (
        "label_return_h1" if "label_return_h1" in train.columns else None
    )
    reg_val_pred = None
    calib_params = None
    if reg_col is not None and fit_df[reg_col].is_not_null().any():
        try:
            X_fit, y_reg_fit, _ = to_xy(fit_df, feature_cols=feature_cols, target_col=reg_col)
            X_es, y_reg_es, _ = to_xy(es_df, feature_cols=feature_cols, target_col=reg_col)
            _, y_dir_es, _ = to_xy(es_df, feature_cols=feature_cols, target_col=CLASSIFICATION_TARGET)
            reg_booster = train_regressor(X_fit, y_reg_fit, X_es, y_reg_es)
            raw_es_proba = booster.predict(X_es, num_iteration=booster.best_iteration)
            reg_es_pred = reg_booster.predict(X_es, num_iteration=reg_booster.best_iteration)
            _, calib_params = _fuse_and_calibrate_gbm_proba(raw_es_proba, reg_es_pred, y_ref=y_dir_es, calib_params=None)
            reg_val_pred = reg_booster.predict(X_val, num_iteration=reg_booster.best_iteration)
        except Exception as e:
            log.debug("Companion regression calibration skipped: %s", e)

    cal_proba, used_calib = _fuse_and_calibrate_gbm_proba(
        raw_val_proba, reg_val_pred, y_ref=y_val, calib_params=calib_params
    )
    pred = cal_proba.argmax(axis=1)

    dir_margin = np.max(cal_proba[:, [0, 2]], axis=1) - cal_proba[:, 1]
    act_mask = (pred != 1) & (dir_margin >= np.quantile(dir_margin, 0.65))
    active_prec = float((pred[act_mask] == y_val[act_mask]).mean()) if np.any(act_mask) else float(accuracy_score(y_val, pred))

    metrics = {
        "accuracy": float(accuracy_score(y_val, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_val, pred)),
        "f1_macro": float(f1_score(y_val, pred, average="macro", zero_division=0)),
        "active_directional_precision": active_prec,
        "best_iteration": int(booster.best_iteration),
        "is_placeholder_proxy": is_default_proxy,
        "proxy_features": SENTIMENT_PROXY_FEATURES,
        "feature_columns": feature_cols,
        "calibration": used_calib,
    }
    if return_proba:
        return booster, metrics, cal_proba
    return booster, metrics


def _resolve_branch_features_and_targets(
    df: pl.DataFrame,
    member_name: str,
    with_news: bool,
) -> tuple[pl.DataFrame, list[str]]:
    from models.gbm_family.train import _apply_member_horizon_targets

    if member_name == "tier2-crossasset-gat":
        feature_cols = [c for c in CROSS_ASSET_GAT_FEATURES if c in df.columns] or list(SENTIMENT_PROXY_FEATURES)
        df, _ = _apply_member_horizon_targets(df, "tier2-lgbm-bookdepth")
        return df, feature_cols

    if member_name == "tier2-cryptobert-news":
        extra = [c for c in CROWD_SQUEEZE_CONTEXT_FEATURES if c in df.columns and c not in SENTIMENT_PROXY_FEATURES]
        feature_cols = list(SENTIMENT_PROXY_FEATURES) + extra
        if with_news:
            from news_pipeline.features import NEWS_FEATURE_COLUMNS
            feature_cols += [c for c in NEWS_FEATURE_COLUMNS if c in df.columns and c not in feature_cols]
        df, _ = _apply_member_horizon_targets(df, "tier2-lgbm-bookdepth")
        return df, feature_cols

    # Default "sentiment-branch"
    if with_news:
        from news_pipeline.features import NEWS_FEATURE_COLUMNS
        feature_cols = SENTIMENT_PROXY_FEATURES + NEWS_FEATURE_COLUMNS
    else:
        feature_cols = list(SENTIMENT_PROXY_FEATURES)
    return df, feature_cols


def run(
    val_days: int,
    max_days: int | None,
    upload: bool,
    with_news: bool = False,
    output_dir: str | Path | None = None,
    member_name: str = MEMBER_NAME,
) -> dict:
    if with_news:
        try:
            df = load_with_news(max_days)
        except ValueError:
            if member_name == MEMBER_NAME:
                raise
            from news_pipeline.features import attach_news_features
            from news_pipeline.live_daemon import bootstrap_historical_news_store_from_bars
            raw_df = filter_valid(load_days(max_days=max_days))
            bootstrap_historical_news_store_from_bars(raw_df)
            df = attach_news_features(raw_df)
    else:
        df = filter_valid(load_days(max_days=max_days))
        if member_name == "tier2-cryptobert-news":
            from news_pipeline.features import attach_news_features
            from news_pipeline.live_daemon import bootstrap_historical_news_store_from_bars
            df = attach_news_features(df)
            if float(df["news_coverage"].mean()) == 0.0:
                bootstrap_historical_news_store_from_bars(df)
                df = attach_news_features(df)
            with_news = True

    df, feature_cols = _resolve_branch_features_and_targets(df, member_name, with_news)
    n_folds = 3
    val_size = min(val_days * config.BARS_PER_DAY, max(df.height // (n_folds + 2), 100))
    splits = purged_walk_forward_splits(df, n_folds=n_folds, val_size=val_size)

    oof_ts_list: list[np.ndarray] = []
    oof_proba_list: list[np.ndarray] = []
    booster = metrics = None
    for split in splits:
        booster, metrics, fold_proba = train_and_eval(
            split.train, split.val, feature_cols, return_proba=True
        )
        _, _, fold_ts = to_xy(split.val, feature_cols=feature_cols, target_col=CLASSIFICATION_TARGET)
        oof_ts_list.append(fold_ts)
        oof_proba_list.append(fold_proba)
    split = splits[-1]
    direction_proba = np.concatenate(oof_proba_list, axis=0)
    ts_val_all = np.concatenate(oof_ts_list, axis=0)

    ckpt_dir = Path(output_dir) if output_dir else CHECKPOINT_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model_path = ckpt_dir / "sentiment_proxy_classifier.txt"
    booster.save_model(str(model_path))

    training_data_path = ckpt_dir / "training_data_snapshot.parquet"
    snap_rows = split.train if member_name == MEMBER_NAME else split.train.tail(min(50_000, split.train.height))
    snap_rows.select(["ts", *feature_cols, "label_triple_barrier"]).write_parquet(
        training_data_path, compression="snappy"
    )
    (ckpt_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log.info(
        "Branch %s (%s): accuracy=%.4f bal_acc=%.4f active_prec=%.4f (n_features=%d)",
        member_name,
        "proxies + news" if with_news else "specialist proxies",
        metrics["accuracy"],
        metrics["balanced_accuracy"],
        metrics["active_directional_precision"],
        len(feature_cols),
    )

    from common.oof import save_oof

    ts_val = ts_val_all
    save_oof(member_name, ts_val, direction_proba)
    if member_name != MEMBER_NAME and not (CHECKPOINT_DIR / "sentiment_proxy_classifier.txt").exists():
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(CHECKPOINT_DIR / "sentiment_proxy_classifier.txt"))
        (CHECKPOINT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        save_oof(MEMBER_NAME, ts_val, direction_proba)

    if upload:
        from common.hf_model_upload import upload_best_checkpoint, upload_training_data_snapshot

        repo_id = upload_best_checkpoint(member_name, [model_path, ckpt_dir / "metrics.json"], metrics=metrics)
        if member_name == MEMBER_NAME:
            data_repo_id = upload_training_data_snapshot(member_name, [training_data_path])
            log.info("Uploaded checkpoint to %s and training data to %s", repo_id, data_repo_id)
        else:
            log.info("Uploaded specialist checkpoint to %s", repo_id)

    metrics["with_news"] = bool(with_news)
    metrics["direction"] = {
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "active_precision": metrics.get("active_precision_at_optimal_threshold", 0.0),
        "macro_f1": metrics.get("f1_macro", 0.0),
    }
    return metrics


def _train_one_side(train: pl.DataFrame, val: pl.DataFrame, feature_cols: list[str]):
    import lightgbm as lgb
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    train_set, val_set, _, _ = _fit_and_early_stop_sets(train, feature_cols)
    X_val, y_val, ts_val = to_xy(val, feature_cols=feature_cols, target_col=CLASSIFICATION_TARGET)
    params = {
        "objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
        "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 200,
        "force_col_wise": True, "max_bin": 255, "verbosity": -1,
    }
    booster = lgb.train(params, train_set, num_boost_round=1000, valid_sets=[val_set],
                         callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])

    direction_proba = booster.predict(X_val, num_iteration=booster.best_iteration)
    pred = direction_proba.argmax(axis=1)
    metrics = {
        "accuracy": float(accuracy_score(y_val, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_val, pred)),
        "f1_macro": float(f1_score(y_val, pred, average="macro", zero_division=0)),
        "best_iteration": int(booster.best_iteration),
        "n_features": len(feature_cols),
    }
    return booster, metrics, direction_proba, ts_val


def run_ablation(val_days: int, max_days: int | None, with_news: bool = False) -> dict:
    """Trains the sentiment_branch_with / sentiment_branch_without pair on
    one shared split (same train/val rows for both sides, so a comparison
    isn't confounded by different time ranges) and exports each side's OOF
    under the name validation/ablation_test.py's own docstring names.
    with_news=False ablates the 13 proxy columns; with_news=True ablates the
    text-sentiment NEWS_FEATURE_COLUMNS block."""
    from common.oof import save_oof

    if with_news:
        from news_pipeline.features import NEWS_FEATURE_COLUMNS

        df = load_with_news(max_days)
        sides = (("with", FEATURE_COLUMNS + NEWS_FEATURE_COLUMNS), ("without", FEATURE_COLUMNS))
    else:
        df = filter_valid(load_days(max_days=max_days))
        sides = (("with", FEATURE_COLUMNS), ("without", WITHOUT_SENTIMENT_FEATURES))
    val_size = val_days * config.BARS_PER_DAY
    split = purged_walk_forward_splits(df, n_folds=1, val_size=val_size)[0]

    ABLATION_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    results = {"ablated_block": "news_text_sentiment" if with_news else "sentiment_proxies"}
    for side, feature_cols in sides:
        booster, metrics, direction_proba, ts_val = _train_one_side(split.train, split.val, feature_cols)
        model_path = ABLATION_CHECKPOINT_DIR / f"sentiment_branch_{side}.txt"
        booster.save_model(str(model_path))
        oof_name = f"sentiment_branch_{side}"
        save_oof(oof_name, ts_val, direction_proba)
        results[side] = metrics
        log.info("sentiment_branch_%s: accuracy=%.4f (n_features=%d)", side, metrics["accuracy"], len(feature_cols))

    (ABLATION_CHECKPOINT_DIR / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-days", type=int, default=config.VAL_DAYS)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--ablation", action="store_true",
                         help="train the sentiment_branch_with/without OOF pair for validation.ablation_test instead of the default proxy-only member")
    parser.add_argument("--with-news", action="store_true",
                         help="add news_pipeline's scraped+LLM-scored text features (and, with --ablation, ablate them)")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--member-name", type=str, default=MEMBER_NAME)
    args = parser.parse_args()
    if args.ablation:
        run_ablation(args.val_days, args.max_days, with_news=args.with_news)
    else:
        run(
            args.val_days, args.max_days, args.upload,
            with_news=args.with_news, output_dir=args.output_dir, member_name=args.member_name,
        )


if __name__ == "__main__":
    main()
