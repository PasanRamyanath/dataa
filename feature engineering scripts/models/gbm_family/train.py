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


NON_STATIONARY_OR_CALENDAR_ID_COLS = {
    "open_interest",
    "open_interest_value",
    "implied_vol_ma_720",
    "premium_index_ma_2880",
    "day_of_month_sin",
    "day_of_month_cos",
    "day_of_week",
    "day_of_week_sin",
    "day_of_week_cos",
    "is_month_start",
    "is_month_end_window",
    "is_quarter_end_window",
}


def _lgb_dataset(X: np.ndarray, y: np.ndarray, feature_cols: list[str] | None = None):
    import lightgbm as lgb

    feature_cols = feature_cols or FEATURE_COLUMNS
    cat_idx = [feature_cols.index(c) for c in CATEGORICAL_OR_FLAG_FEATURE_COLUMNS if c in feature_cols]
    return lgb.Dataset(X, label=y, feature_name=list(feature_cols), categorical_feature=cat_idx, free_raw_data=False)


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
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_data_in_leaf": 250,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 1.0,
        "lambda_l2": 2.0,
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


def _get_member_feature_columns(base_cols: list[str], member_name: str) -> list[str]:
    """
    Subsets the full feature list to orthogonal, stationary features tailored
    to each ensemble member's specific structural mechanism (features/orthogonal_features.py).
    """
    f1_leadlag = {
        "eth_lead_return_lag_1", "eth_lead_return_lag_2", "eth_lead_return_lag_3",
        "cross_return_spread_10s", "cross_return_spread_30s", "cross_return_residual_10s",
        "cross_ofi_divergence_10s", "cross_ofi_divergence_30s", "cross_mom_diff_30s",
        "cross_mom_diff_zscore_30s", "eth_lead_ret_bps", "eth_imbalance",
        "toptrader_sum_ls_ratio", "taker_ls_vol_ratio", "all_account_ls_ratio",
    }
    f2_basis_funding = {
        "basis_spread_bps", "basis_zscore_180", "basis_zscore_720",
        "basis_velocity_10s", "basis_velocity_30s", "basis_acceleration_30s",
        "funding_pressure_bps", "funding_basis_divergence", "funding_rate_pressure",
        "funding_rate", "funding_rate_ma_8h", "funding_rate_ma_24h", "funding_rate_change",
        "funding_oi_crowding_180", "funding_oi_crowding_720", "open_interest_pct_change",
        "open_interest_zscore_30", "open_interest_zscore_180", "toptrader_count_ls_ratio",
        "toptrader_sum_ls_ratio", "all_account_ls_ratio", "taker_ls_vol_ratio",
        "implied_vol", "iv_realized_vol_spread",
    }
    f3_microstructure = {
        "l1_queue_imbalance", "l1_queue_imbalance_ma3", "l1_queue_velocity",
        "microprice_drift_bps", "microprice_velocity_10s", "microprice_velocity_30s",
        "trade_flow_imbalance", "tfi_ma_3", "vpin_toxicity_6", "vpin_surge_ratio",
        "liq_net_flow_bps", "liq_burst_zscore", "l1_imbalance", "vpin", "ofi",
        "bids_depth", "asks_depth", "depth_imbalance", "spread_bps", "order_flow_imbalance",
    }

    if member_name == "tier3-lgbm-positioning":
        cols = [c for c in base_cols if c in f2_basis_funding or "funding" in c or "oi" in c or "interest" in c or "basis" in c or "ratio" in c or "vol" in c]
        return cols if len(cols) >= 15 else base_cols
    elif member_name == "tier2-lgbm-bookdepth":
        cols = [c for c in base_cols if c in f3_microstructure or "depth" in c or "imbalance" in c or "flow" in c or "spread" in c or "vpin" in c or "queue" in c]
        return cols if len(cols) >= 15 else base_cols
    elif member_name == "tier1-spot1s-leadlag":
        cols = [c for c in base_cols if c in f1_leadlag or "eth" in c or "lead" in c or "lag" in c or "spread" in c or "basis" in c]
        return cols if len(cols) >= 10 else base_cols
    elif member_name == "tier1-queue-hazard":
        cols = [c for c in base_cols if c in f3_microstructure or "hazard" in c or "queue" in c or "vpin" in c or "drift" in c]
        return cols if len(cols) >= 10 else base_cols
    elif member_name == "tier1-lgbm-micro-1s10s":
        cols = [c for c in base_cols if c in f3_microstructure or "10s" in c or "velocity" in c or "imbalance" in c or "return_1" in c]
        return cols if len(cols) >= 15 else base_cols
    return base_cols


def _apply_member_horizon_targets(df: pl.DataFrame, member_name: str) -> tuple[pl.DataFrame, str, dict]:
    """
    Configures horizon-specific Fee-Floored Triple-Barrier / Return targets and
    regularization parameters per new_plan.md §6.2, §23.1, and §24 so each tier's
    LightGBM specialist trains on its authentic operational horizon rather than
    duplicating the 10-second H1 model.
    """
    has_ohlc = ("close" in df.columns and "high" in df.columns and "low" in df.columns)
    if has_ohlc:
        try:
            from labeling_multi_horizon_v4 import fee_floored_triple_barrier_labels_v4
            if member_name == "tier3-lgbm-positioning":
                df = fee_floored_triple_barrier_labels_v4(
                    df, max_hold_bars=12, pt_mult=1.85, sl_mult=1.15, min_pt_bps=24.0, min_sl_bps=16.0,
                    maker_tp_fee_bps=4.0, passive_unwind_fee_bps=5.0, taker_sl_fee_bps=7.5, prefix="meso_15m",
                )
                dir_col = "label_meso_15m_tb"
                reg_col = "label_meso_15m_net_long_bps"
                df = df.with_columns([
                    (pl.col(dir_col) != 0).cast(pl.Int8).alias(META_LABEL_TARGET),
                    pl.col(dir_col).alias(CLASSIFICATION_TARGET),
                ])
                return df, reg_col, {"num_leaves": 31, "max_depth": 6, "min_data_in_leaf": 300, "learning_rate": 0.03}

            elif member_name == "tier2-lgbm-bookdepth":
                df = fee_floored_triple_barrier_labels_v4(
                    df, max_hold_bars=6, pt_mult=1.5, sl_mult=1.0, min_pt_bps=14.0, min_sl_bps=10.0,
                    maker_tp_fee_bps=4.0, passive_unwind_fee_bps=5.0, taker_sl_fee_bps=7.5, prefix="tactical_5m",
                )
                dir_col = "label_tactical_5m_tb"
                reg_col = "label_tactical_5m_net_long_bps"
                df = df.with_columns([
                    (pl.col(dir_col) != 0).cast(pl.Int8).alias(META_LABEL_TARGET),
                    pl.col(dir_col).alias(CLASSIFICATION_TARGET),
                ])
                return df, reg_col, {"num_leaves": 31, "max_depth": 6, "min_data_in_leaf": 250, "learning_rate": 0.03}

            elif member_name in ("tier1-spot1s-leadlag", "tier1-queue-hazard", "tier1-lgbm-micro-1s10s"):
                df = fee_floored_triple_barrier_labels_v4(
                    df, max_hold_bars=3, pt_mult=1.2, sl_mult=1.0, min_pt_bps=6.0, min_sl_bps=5.0,
                    maker_tp_fee_bps=4.0, passive_unwind_fee_bps=5.0, taker_sl_fee_bps=7.5, prefix="micro_10s",
                )
                dir_col = "label_micro_10s_tb"
                reg_col = "label_micro_10s_net_long_bps"
                df = df.with_columns([
                    (pl.col(dir_col) != 0).cast(pl.Int8).alias(META_LABEL_TARGET),
                    pl.col(dir_col).alias(CLASSIFICATION_TARGET),
                ])
                return df, reg_col, {"num_leaves": 28, "max_depth": 5, "min_data_in_leaf": 250, "learning_rate": 0.035}
        except Exception as exc:
            log.warning("fee_floored_triple_barrier_labels_v4 computation failed: %s; using return deadbands", exc)

    if member_name == "tier3-lgbm-positioning" and "label_return_h4" in df.columns:
        # Tier 3 Meso (label_return_h4): fee-floored deadband = 18.0 bps (0.0018)
        deadband = pl.lit(0.0018)
        ret = pl.col("label_return_h4").fill_nan(0.0).fill_null(0.0)
        df = df.with_columns([
            pl.when(ret > deadband).then(pl.lit(1, dtype=pl.Int8))
            .when(ret < -deadband).then(pl.lit(-1, dtype=pl.Int8))
            .otherwise(pl.lit(0, dtype=pl.Int8))
            .alias(CLASSIFICATION_TARGET),
            (ret.abs() > deadband).cast(pl.Int8).alias(META_LABEL_TARGET),
            ret.cast(pl.Float32).alias("label_return_h4"),
        ])
        return df, "label_return_h4", {"num_leaves": 31, "max_depth": 6, "min_data_in_leaf": 300, "learning_rate": 0.03}

    if member_name == "tier2-lgbm-bookdepth" and "label_return_h3" in df.columns:
        # Tier 2 Tactical (label_return_h3): fee-floored deadband = 14.0 bps (0.0014)
        deadband = pl.lit(0.0014)
        ret = pl.col("label_return_h3").fill_nan(0.0).fill_null(0.0)
        df = df.with_columns([
            pl.when(ret > deadband).then(pl.lit(1, dtype=pl.Int8))
            .when(ret < -deadband).then(pl.lit(-1, dtype=pl.Int8))
            .otherwise(pl.lit(0, dtype=pl.Int8))
            .alias(CLASSIFICATION_TARGET),
            (ret.abs() > deadband).cast(pl.Int8).alias(META_LABEL_TARGET),
            ret.cast(pl.Float32).alias("label_return_h3"),
        ])
        return df, "label_return_h3", {"num_leaves": 31, "max_depth": 6, "min_data_in_leaf": 250, "learning_rate": 0.03}

    if member_name == "tier1-spot1s-leadlag" and "label_return_h2" in df.columns:
        # Tier 1 Micro/Lead-Lag (label_return_h2): deadband = 6.5 bps (0.00065)
        deadband = pl.lit(0.00065)
        ret = pl.col("label_return_h2").fill_nan(0.0).fill_null(0.0)
        df = df.with_columns([
            pl.when(ret > deadband).then(pl.lit(1, dtype=pl.Int8))
            .when(ret < -deadband).then(pl.lit(-1, dtype=pl.Int8))
            .otherwise(pl.lit(0, dtype=pl.Int8))
            .alias(CLASSIFICATION_TARGET),
            (ret.abs() > deadband).cast(pl.Int8).alias(META_LABEL_TARGET),
            ret.cast(pl.Float32).alias("label_return_h2"),
        ])
        return df, "label_return_h2", {"num_leaves": 31, "max_depth": 5, "min_data_in_leaf": 250, "learning_rate": 0.035}

    if member_name == "tier1-queue-hazard" and "label_return_h1" in df.columns:
        # Tier 1 Queue Depletion & Safe Passive Fill Hazard (10s-30s)
        ret1 = pl.col("label_return_h1").fill_nan(0.0).fill_null(0.0)
        ret2 = pl.col("label_return_h2").fill_nan(0.0).fill_null(0.0) if "label_return_h2" in df.columns else ret1
        composite_ret = 0.6 * ret1 + 0.4 * ret2
        deadband = pl.lit(0.00035)
        df = df.with_columns([
            pl.when(composite_ret > deadband).then(pl.lit(1, dtype=pl.Int8))
            .when(composite_ret < -deadband).then(pl.lit(-1, dtype=pl.Int8))
            .otherwise(pl.lit(0, dtype=pl.Int8))
            .alias(CLASSIFICATION_TARGET),
            (composite_ret.abs() > deadband).cast(pl.Int8).alias(META_LABEL_TARGET),
        ])
        return df, REGRESSION_TARGET, {"num_leaves": 24, "max_depth": 5, "min_data_in_leaf": 300, "learning_rate": 0.03}

    if "label_return_h1" in df.columns:
        # Tier 1 Micro 10s (tier1-lgbm-micro-1s10s / gbm-family): fee-floored 3.0 bps deadband (0.0003)
        ret1 = pl.col("label_return_h1").fill_nan(0.0).fill_null(0.0)
        deadband = pl.lit(0.00030)
        df = df.with_columns([
            pl.when(ret1 > deadband).then(pl.lit(1, dtype=pl.Int8))
            .when(ret1 < -deadband).then(pl.lit(-1, dtype=pl.Int8))
            .otherwise(pl.lit(0, dtype=pl.Int8))
            .alias(CLASSIFICATION_TARGET),
            (ret1.abs() > deadband).cast(pl.Int8).alias(META_LABEL_TARGET),
        ])
    return df, REGRESSION_TARGET, {"num_leaves": 31, "max_depth": 6, "min_data_in_leaf": 250, "learning_rate": 0.03}


def run(
    n_folds: int,
    val_days: int,
    max_days: int | None,
    upload: bool,
    with_news: bool = False,
    member_name: str = "gbm-family",
    output_dir: Path | None = None,
) -> dict:
    if with_news:
        # §9.3 route (1): text-sentiment features as extra GBM inputs. Keep
        # it only if validation.ablation_test shows it clears §10.3's bar.
        from models.sentiment_branch.train import load_with_news
        from news_pipeline.features import NEWS_FEATURE_COLUMNS

        df = load_with_news(max_days)
        base_cols = [c for c in FEATURE_COLUMNS if c not in NON_STATIONARY_OR_CALENDAR_ID_COLS]
        feature_cols = base_cols + NEWS_FEATURE_COLUMNS
    else:
        df = filter_valid(load_days(max_days=max_days))
        feature_cols = [c for c in FEATURE_COLUMNS if c not in NON_STATIONARY_OR_CALENDAR_ID_COLS]

    df, reg_target_col, extra_params = _apply_member_horizon_targets(df, member_name)
    feature_cols = _get_member_feature_columns(feature_cols, member_name)
    val_size = val_days * config.BARS_PER_DAY
    splits = purged_walk_forward_splits(df, n_folds=n_folds, val_size=val_size)

    cv_report = {"folds": []}
    best_dir_booster, best_dir_metrics, best_dir_score = None, None, -1.0
    best_meta_booster, best_meta_metrics, best_meta_score = None, None, -1.0
    best_reg_booster, best_reg_metrics, best_reg_score = None, None, float("inf")

    for split in splits:
        d_b, direction_metrics = train_classifier(split.train, split.val, num_class=3, target=CLASSIFICATION_TARGET,
                                                  params_extra=extra_params, feature_cols=feature_cols)
        m_b, meta_metrics = train_classifier(split.train, split.val, num_class=2, target=META_LABEL_TARGET,
                                             params_extra=extra_params, feature_cols=feature_cols)
        r_b, regression_metrics = train_regressor(split.train, split.val, target=reg_target_col, feature_cols=feature_cols)

        d_score = direction_metrics["balanced_accuracy"] + (0.05 if direction_metrics["best_iteration"] >= 5 else 0.0)
        if d_score >= best_dir_score:
            best_dir_score = d_score
            best_dir_booster, best_dir_metrics = d_b, direction_metrics

        m_score = meta_metrics["balanced_accuracy"] + (0.05 if meta_metrics["best_iteration"] >= 5 else 0.0)
        if m_score >= best_meta_score:
            best_meta_score = m_score
            best_meta_booster, best_meta_metrics = m_b, meta_metrics

        r_score = regression_metrics["mae"] - (0.001 if regression_metrics["best_iteration"] >= 5 else 0.0)
        if r_score <= best_reg_score:
            best_reg_score = r_score
            best_reg_booster, best_reg_metrics = r_b, regression_metrics

        fold_report = {
            "fold_index": split.fold_index,
            "train_rows": split.train.height,
            "val_rows": split.val.height,
            "direction": direction_metrics,
            "meta_label": meta_metrics,
            "regression_h1": regression_metrics,
        }
        cv_report["folds"].append(fold_report)
        log.info("[%s] Fold %d: direction_acc=%.4f (bal=%.4f, it=%d) meta_acc=%.4f reg_mae=%.6f",
                  member_name, split.fold_index, direction_metrics["accuracy"],
                  direction_metrics["balanced_accuracy"], direction_metrics["best_iteration"],
                  meta_metrics["accuracy"], regression_metrics["mae"])

    final_split = splits[-1]
    direction_booster, direction_metrics = best_dir_booster, best_dir_metrics
    meta_booster, meta_metrics = best_meta_booster, best_meta_metrics
    regression_booster, regression_metrics = best_reg_booster, best_reg_metrics

    target_chk_dir = Path(output_dir) if output_dir else CHECKPOINT_DIR
    target_chk_dir.mkdir(parents=True, exist_ok=True)
    direction_path = target_chk_dir / "direction_classifier.txt"
    meta_path = target_chk_dir / "meta_label_classifier.txt"
    regression_path = target_chk_dir / "return_h1_regressor.txt"
    direction_booster.save_model(str(direction_path))
    meta_booster.save_model(str(meta_path))
    regression_booster.save_model(str(regression_path))

    full_report = {
        "member_name": member_name,
        "regression_target": reg_target_col,
        "cv": cv_report,
        "final_fold": {
            "direction": direction_metrics,
            "meta_label": meta_metrics,
            "regression_h1": regression_metrics,
        },
        "feature_columns": feature_cols,
        "with_news": with_news,
    }
    (target_chk_dir / "metrics.json").write_text(json.dumps(full_report, indent=2), encoding="utf-8")
    log.info("Saved %s checkpoints to %s", member_name, target_chk_dir)

    from common.oof import save_oof

    X_val_final, _, ts_val_final = to_xy(final_split.val, feature_cols=feature_cols, target_col=CLASSIFICATION_TARGET)
    direction_proba = direction_booster.predict(X_val_final, num_iteration=direction_booster.best_iteration)
    save_oof(member_name, ts_val_final, direction_proba)
    if member_name in ("gbm-family", "tier1-lgbm-micro-1s10s"):
        save_oof("gbm-family", ts_val_final, direction_proba)

    if upload and member_name == "gbm-family":
        from common.hf_model_upload import upload_best_checkpoint, upload_training_data_snapshot

        training_data_path = target_chk_dir / "training_data_snapshot.parquet"
        final_split.train.write_parquet(training_data_path, compression="snappy")
        repo_id = upload_best_checkpoint(
            "gbm-family",
            [direction_path, meta_path, regression_path, target_chk_dir / "metrics.json"],
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
    parser.add_argument("--member-name", type=str, default="gbm-family")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()
    run(
        args.n_folds,
        args.val_days,
        args.max_days,
        args.upload,
        with_news=args.with_news,
        member_name=args.member_name,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )


if __name__ == "__main__":
    main()
