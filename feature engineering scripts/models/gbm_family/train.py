"""
GBM family ensemble member: gradient-boosted trees on the full 184-column
tabular feature set (common/feature_contract.py), trained on three targets:

  - direction (label_triple_barrier, 3-class: down/flat/up)
  - meta-label (label_meta_act, binary: will a profit-take/stop-loss barrier be touched before the time barrier, either side)
  - short-horizon return magnitude (label_return_h1, regression)

GBMs are the one ensemble member that consumes the tabular feature set
directly (no lookback windowing) and natively handles the mixed
numeric/categorical/flag columns in CATEGORICAL_OR_FLAG_FEATURE_COLUMNS as
LightGBM categorical features — every sequence model in models/{tft,
patchtst,itransformer,nhits,ssm_challenger,kronos_finetune}/ instead needs a
fixed-length lookback window built from the same feature contract.

Evaluated via the exact purged/embargoed walk-forward folds every other
member uses (common.cv), then a final model is refit on all data up to the
last fold's embargo boundary (never touching the final held-out val window)
and uploaded, together with the exact parquet it was trained on, to private
HF repos (common.hf_model_upload).

--with-news appends news_pipeline's NEWS_FEATURE_COLUMNS (scraped articles,
two-tier CryptoBERT + LLM scored, point-in-time multi-window aggregates);
metrics.json["feature_columns"] records the list so the live loader
attaches the same block at inference.

Usage:
    python -m models.gbm_family.train --n-folds 5 --val-days 7 --upload
    python -m models.gbm_family.train --n-folds 5 --val-days 7 --with-news
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
from common.feature_contract import (
    CATEGORICAL_OR_FLAG_FEATURE_COLUMNS,
    CLASSIFICATION_TARGET,
    FEATURE_COLUMNS,
    META_LABEL_TARGET,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("models.gbm_family.train")

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
REGRESSION_TARGET = "label_return_h1"  # primary short-horizon target for this member; h2-h4 left to sequence members


def _lgb_dataset(X: np.ndarray, y: np.ndarray, feature_cols: list[str] | None = None):
    import lightgbm as lgb

    feature_cols = feature_cols or FEATURE_COLUMNS
    cat_idx = [feature_cols.index(c) for c in CATEGORICAL_OR_FLAG_FEATURE_COLUMNS if c in feature_cols]
    return lgb.Dataset(X, label=y, categorical_feature=cat_idx, free_raw_data=False)


EARLY_STOP_FRACTION = 0.1


def split_early_stopping(train: pl.DataFrame, fraction: float = EARLY_STOP_FRACTION,
                         purge_bars: int = config.PURGE_BARS) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(fit, early_stop): the tail `fraction` of the training window, behind a
    `purge_bars` gap so no fit-row label reaches into it, picks the boosting
    round count. The fold's val window is then touched exactly once, for
    scoring — early-stopping on val itself would select the round count on
    the very rows the reported metrics (and the fusion OOF) come from."""
    n_es = int(train.height * fraction)
    n_fit = train.height - n_es - purge_bars
    if n_es < 200 or n_fit < 1000:
        raise ValueError(f"training window of {train.height} rows too small for an early-stopping holdout")
    return train.slice(0, n_fit), train.slice(train.height - n_es, n_es)


def train_classifier(train: pl.DataFrame, val: pl.DataFrame, num_class: int, target: str, params_extra: dict | None = None,
                     feature_cols: list[str] | None = None):
    import lightgbm as lgb

    feature_cols = feature_cols or FEATURE_COLUMNS
    fit, early_stop = split_early_stopping(train)
    X_train, y_train, _ = to_xy(fit, feature_cols=feature_cols, target_col=target)
    X_es, y_es, _ = to_xy(early_stop, feature_cols=feature_cols, target_col=target)
    X_val, y_val, _ = to_xy(val, feature_cols=feature_cols, target_col=target)

    params = {
        "objective": "multiclass" if num_class > 2 else "binary",
        "num_class": num_class if num_class > 2 else None,
        "metric": "multi_logloss" if num_class > 2 else "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
    }
    params = {k: v for k, v in params.items() if v is not None}
    if params_extra:
        params.update(params_extra)

    booster = lgb.train(
        params,
        _lgb_dataset(X_train, y_train, feature_cols),
        num_boost_round=2000,
        valid_sets=[_lgb_dataset(X_es, y_es, feature_cols)],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )

    pred = booster.predict(X_val, num_iteration=booster.best_iteration)
    pred_label = pred.argmax(axis=1) if num_class > 2 else (pred > 0.5).astype(int)

    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    metrics = {
        "accuracy": float(accuracy_score(y_val, pred_label)),
        "balanced_accuracy": float(balanced_accuracy_score(y_val, pred_label)),
        "f1_macro": float(f1_score(y_val, pred_label, average="macro", zero_division=0)),
        "best_iteration": int(booster.best_iteration),
    }
    return booster, metrics


def train_regressor(train: pl.DataFrame, val: pl.DataFrame, target: str, feature_cols: list[str] | None = None):
    import lightgbm as lgb

    feature_cols = feature_cols or FEATURE_COLUMNS
    fit, early_stop = split_early_stopping(train)
    X_train, y_train, _ = to_xy(fit, feature_cols=feature_cols, target_col=target)
    X_es, y_es, _ = to_xy(early_stop, feature_cols=feature_cols, target_col=target)
    X_val, y_val, _ = to_xy(val, feature_cols=feature_cols, target_col=target)

    params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
    }
    booster = lgb.train(
        params,
        _lgb_dataset(X_train, y_train, feature_cols),
        num_boost_round=2000,
        valid_sets=[_lgb_dataset(X_es, y_es, feature_cols)],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    pred = booster.predict(X_val, num_iteration=booster.best_iteration)

    from sklearn.metrics import mean_absolute_error, mean_squared_error

    metrics = {
        "mae": float(mean_absolute_error(y_val, pred)),
        "rmse": float(mean_squared_error(y_val, pred) ** 0.5),
        "best_iteration": int(booster.best_iteration),
    }
    return booster, metrics


def run(n_folds: int, val_days: int, max_days: int | None, upload: bool, with_news: bool = False) -> dict:
    if with_news:
        # §9.3 route (1): text-sentiment features as extra GBM inputs. Keep
        # it only if validation.ablation_test shows it clears §10.3's bar.
        from models.sentiment_branch.train import load_with_news
        from news_pipeline.features import NEWS_FEATURE_COLUMNS

        df = load_with_news(max_days)
        feature_cols = FEATURE_COLUMNS + NEWS_FEATURE_COLUMNS
    else:
        df = filter_valid(load_days(max_days=max_days))
        feature_cols = FEATURE_COLUMNS
    val_size = val_days * config.BARS_PER_DAY
    splits = purged_walk_forward_splits(df, n_folds=n_folds, val_size=val_size)

    cv_report = {"folds": []}
    for split in splits:
        _, direction_metrics = train_classifier(split.train, split.val, num_class=3, target=CLASSIFICATION_TARGET,
                                                feature_cols=feature_cols)
        _, meta_metrics = train_classifier(split.train, split.val, num_class=2, target=META_LABEL_TARGET,
                                           feature_cols=feature_cols)
        _, regression_metrics = train_regressor(split.train, split.val, target=REGRESSION_TARGET, feature_cols=feature_cols)
        fold_report = {
            "fold_index": split.fold_index,
            "train_rows": split.train.height,
            "val_rows": split.val.height,
            "direction": direction_metrics,
            "meta_label": meta_metrics,
            "regression_h1": regression_metrics,
        }
        cv_report["folds"].append(fold_report)
        log.info("Fold %d: direction_acc=%.4f meta_acc=%.4f h1_mae=%.6f",
                  split.fold_index, direction_metrics["accuracy"], meta_metrics["accuracy"], regression_metrics["mae"])

    # Final fit: train on everything up to the LAST fold's embargo boundary
    # (never touching that fold's val window), evaluate once more on it,
    # and that's the checkpoint that gets uploaded.
    final_split = splits[-1]
    direction_booster, direction_metrics = train_classifier(final_split.train, final_split.val, 3, CLASSIFICATION_TARGET,
                                                            feature_cols=feature_cols)
    meta_booster, meta_metrics = train_classifier(final_split.train, final_split.val, 2, META_LABEL_TARGET,
                                                  feature_cols=feature_cols)
    regression_booster, regression_metrics = train_regressor(final_split.train, final_split.val, REGRESSION_TARGET,
                                                             feature_cols=feature_cols)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    direction_path = CHECKPOINT_DIR / "direction_classifier.txt"
    meta_path = CHECKPOINT_DIR / "meta_label_classifier.txt"
    regression_path = CHECKPOINT_DIR / "return_h1_regressor.txt"
    direction_booster.save_model(str(direction_path))
    meta_booster.save_model(str(meta_path))
    regression_booster.save_model(str(regression_path))

    training_data_path = CHECKPOINT_DIR / "training_data_snapshot.parquet"
    final_split.train.write_parquet(training_data_path, compression="snappy")

    full_report = {
        "cv": cv_report,
        "final_fold": {
            "direction": direction_metrics,
            "meta_label": meta_metrics,
            "regression_h1": regression_metrics,
        },
        "feature_columns": feature_cols,
        "with_news": with_news,
    }
    (CHECKPOINT_DIR / "metrics.json").write_text(json.dumps(full_report, indent=2), encoding="utf-8")
    log.info("Saved checkpoints to %s", CHECKPOINT_DIR)

    from common.oof import save_oof

    X_val_final, _, ts_val_final = to_xy(final_split.val, feature_cols=feature_cols, target_col=CLASSIFICATION_TARGET)
    direction_proba = direction_booster.predict(X_val_final, num_iteration=direction_booster.best_iteration)
    save_oof("gbm-family", ts_val_final, direction_proba)

    if upload:
        from common.hf_model_upload import upload_best_checkpoint, upload_training_data_snapshot

        repo_id = upload_best_checkpoint(
            "gbm-family",
            [direction_path, meta_path, regression_path, CHECKPOINT_DIR / "metrics.json"],
            metrics=full_report,
        )
        data_repo_id = upload_training_data_snapshot("gbm-family", [training_data_path])
        log.info("Uploaded checkpoint to %s and training data to %s", repo_id, data_repo_id)

    return full_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--val-days", type=int, default=config.VAL_DAYS)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--with-news", action="store_true",
                        help="append news_pipeline's point-in-time text-sentiment features to the 184 FEATURE_COLUMNS")
    args = parser.parse_args()
    run(args.n_folds, args.val_days, args.max_days, args.upload, with_news=args.with_news)


if __name__ == "__main__":
    main()
