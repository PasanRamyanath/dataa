"""
GBM-family ensemble member: LightGBM 3-class direction classifier (over
triple_barrier_labels), meta-labeling binary classifier (act / avoid over
add_meta_label), and multi-horizon return regressor, trained across purged
walk-forward splits with embargo.

Key v4.0 Upgrades for Peak Accuracy, Precision & Kaggle 30GB RAM Efficiency:
  1. Orthogonal Domain-Specific Feature Subspaces per Specialist:
     - `tier3-lgbm-positioning`: Positioning, Open Interest, Funding, Term Structure & Meso Trend (`h3 = 1h`).
     - `tier2-lgbm-bookdepth`: Order-Book Depth (`0.2%..5%`), Multi-Level OFI, Spread & Tactical Reversion (`h2 = 10m`).
     - `tier1-lgbm-micro-1s10s`: Microstructure `10s/1m` Velocity, Trade Intensity, Kyle's Lambda, Roll Spread & VPIN (`h1 = 2m`).
     - `tier1-spot1s-leadlag`: Spot-Perp Lead-Lag, Basis Z-Score, Premium Index & Bybit Cross-Exchange Flow (`h1 = 2m`).
     - `tier1-queue-hazard`: BBO Queue Depletion, Spread Regime, Trade Size Surge & Non-Adverse Fill Survival (`h1 = 2m`).
     - `gbm-family`: Full canonical stationary feature contract.
     Every specialist's `feature_columns` is strictly a subset of `FEATURE_COLUMNS` (plus `news_*` when `--with-news`),
     guaranteeing 100% live inference compatibility with `checkpoint_loaders.py`!
  2. Volatility-Normalized Fee-Floored Targets & SNR Sample Weighting:
     Adaptive deadbands scaled by trailing realized volatility (`realized_vol_90`) and
     Signal-to-Noise Ratio (SNR) + Class-Balanced sample weights so tree splits focus
     on decisive barrier resolutions rather than borderline noise.
  3. Dual-Head Classifier + Huber Return Regressor Probability Fusion & Prior Debiasing:
     Combines debiased multiclass probabilities with the continuous return regressor's
     implied directional probabilities to maximize validation accuracy and active precision.
  4. Memory & Disk Isolation:
     Uses `force_col_wise=True`, `max_bin=255`, bounded CPU threads, and isolated
     checkpoint directories without redundant 875 MB snapshot writes per sub-member.

Usage:
    python -m models.gbm_family.train --n-folds 3 --val-days 7 --upload
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
from pathlib import Path

import numpy as np
import polars as pl

import config
from common.cv import purged_walk_forward_splits
from common.data import filter_valid, load_days, to_xy
from common.feature_contract import (
    CLASSIFICATION_TARGET,
    FEATURE_COLUMNS,
    META_LABEL_TARGET,
    REGRESSION_TARGETS,
)
from common.oof import regression_to_direction_proba

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("models.gbm_family.train")

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"


def _select_specialist_features(member_name: str, base_cols: list[str]) -> list[str]:
    """
    Selects the domain-specific feature subspace (plus core multi-scale volatility/momentum
    anchors) for each specialist member from `base_cols` (strictly preserving contract column
    names so live inference `recent_df.select(feature_cols)` always succeeds).
    """
    if member_name == "gbm-family" or len(base_cols) <= 35:
        return list(base_cols)

    core_anchors = {
        "log_return_1", "log_return_3", "log_return_6", "log_return_18", "log_return_36",
        "log_return_90", "log_return_180", "log_return_360",
        "realized_vol_15", "realized_vol_90", "realized_vol_360",
        "gk_vol_90", "parkinson_vol_90", "vol_of_vol_90",
        "vwap_deviation", "bollinger_pct_b", "rsi_14", "rsi_30",
        "adx_14", "atr_pct_14", "donchian_pos_180", "hurst_180",
        "htf_1m_return", "htf_5m_return",
    }

    if member_name == "tier3-lgbm-positioning":
        keywords = (
            "funding_", "open_interest", "oi_", "toptrader_", "all_account_", "taker_ls_",
            "basis_", "premium_", "implied_vol", "iv_", "roll_yield", "term_structure",
            "momentum_", "return_360", "return_720", "realized_vol_", "donchian_", "adx_",
            "hurst_", "variance_ratio", "autocorr_", "htf_", "hour_", "session_",
        )
    elif member_name == "tier2-lgbm-bookdepth":
        keywords = (
            "depth_", "ofi_", "order_flow_", "book_", "spread", "taker_buy_sell",
            "volume_delta", "volume_zscore", "vwap_", "bollinger_", "rsi_",
            "breakout_", "dist_to_", "htf_", "log_return_", "realized_vol_",
        )
    elif member_name == "tier1-spot1s-leadlag":
        keywords = (
            "spot_", "basis_", "premium_", "bybit_", "cross_exch_",
            "ofi_", "order_flow_", "taker_buy_sell", "volume_delta",
            "log_return_", "htf_1m_", "vwap_", "realized_vol_",
        )
    elif member_name == "tier1-queue-hazard":
        keywords = (
            "spread", "depth_0.2", "depth_0.5", "depth_1", "ofi_level_0.2", "ofi_level_0.5",
            "order_flow_", "vpin", "kyle_lambda", "amihud", "roll_spread", "corwin_schultz",
            "trade_count", "avg_trade", "large_trade", "taker_buy_sell", "volume_",
            "intra_bar_", "hl_range", "close_pos_in_bar", "log_return_1", "log_return_3",
        )
    elif member_name == "tier1-lgbm-micro-1s10s":
        keywords = (
            "log_return_1", "log_return_2", "log_return_3", "log_return_6", "log_return_12",
            "ofi_", "order_flow_", "taker_buy_sell", "volume_", "trade_count", "avg_trade",
            "large_trade", "vpin", "kyle_lambda", "roll_spread", "hl_range", "body_to_range",
            "upper_wick", "lower_wick", "close_pos_in_bar", "vwap_", "rsi_", "stoch_",
        )
    else:
        return list(base_cols)

    selected = [
        c for c in base_cols
        if c in core_anchors or any(k in c for k in keywords) or c.startswith("news_")
    ]
    return selected if len(selected) >= 20 else list(base_cols)


def _apply_member_horizon_targets(df: pl.DataFrame, member_name: str) -> tuple[pl.DataFrame, str]:
    """
    Aligns each LightGBM specialist's classification target (`label_triple_barrier`),
    meta-label (`label_meta_act`), and regression target (`reg_target`) with its
    designated tier horizon and volatility-normalized fee-floored deadband.
    """
    horizon_map = {
        "tier3-lgbm-positioning": ("label_return_h3", 360.0, 0.38, 0.00045),
        "tier2-lgbm-bookdepth":   ("label_return_h2", 60.0,  0.34, 0.00024),
        "tier1-lgbm-micro-1s10s": ("label_return_h1", 12.0,  0.32, 0.00014),
        "tier1-spot1s-leadlag":   ("label_return_h1", 12.0,  0.28, 0.00012),
        "tier1-queue-hazard":     ("label_return_h1", 12.0,  0.24, 0.00010),
        "gbm-family":             ("label_return_h1", 12.0,  0.30, 0.00014),
    }
    ret_col, horizon_bars, vol_mult, min_deadband = horizon_map.get(
        member_name, ("label_return_h1", 12.0, 0.30, 0.00014)
    )
    if ret_col not in df.columns:
        return df, REGRESSION_TARGETS[0]

    ret_arr = np.nan_to_num(df[ret_col].to_numpy().astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if "realized_vol_90" in df.columns:
        rvol = np.nan_to_num(df["realized_vol_90"].to_numpy().astype(np.float64), nan=0.00025)
    else:
        rvol = np.full(len(ret_arr), 0.00025, dtype=np.float64)

    sigma_h = np.maximum(rvol * np.sqrt(horizon_bars), min_deadband)
    deadband = np.clip(vol_mult * sigma_h, min_deadband, min_deadband * 6.0)

    if member_name == "gbm-family" and "label_triple_barrier" in df.columns:
        tb_refined = df["label_triple_barrier"].fill_null(0).to_numpy().astype(np.int8)
        if META_LABEL_TARGET in df.columns:
            meta_act = df[META_LABEL_TARGET].fill_null(0).to_numpy().astype(np.int8)
        else:
            meta_act = (tb_refined != 0).astype(np.int8)
    else:
        tb_refined = np.where(
            ret_arr > deadband,
            1,
            np.where(ret_arr < -deadband, -1, 0),
        ).astype(np.int8)
        meta_act = (np.abs(ret_arr) > deadband * 0.90).astype(np.int8)

    snr_weight = (0.60 + np.clip(np.abs(ret_arr) / np.maximum(deadband, 1e-6), 0.0, 3.0)).astype(np.float32)

    df = df.with_columns([
        pl.Series(CLASSIFICATION_TARGET, tb_refined, dtype=pl.Int8),
        pl.Series(META_LABEL_TARGET, meta_act, dtype=pl.Int8),
        pl.Series("_snr_weight", snr_weight, dtype=pl.Float32),
    ])
    return df, ret_col


def _compute_balanced_snr_weights(df_slice: pl.DataFrame, y: np.ndarray) -> np.ndarray:
    counts = np.bincount(np.clip(y.astype(np.int64), 0, 2), minlength=3).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv_freq = counts.sum() / (len(counts) * counts)
    class_w = inv_freq[np.clip(y.astype(np.int64), 0, 2)]
    if "_snr_weight" in df_slice.columns:
        snr_w = np.nan_to_num(df_slice["_snr_weight"].to_numpy().astype(np.float64), nan=1.0)
    else:
        snr_w = np.ones_like(class_w)
    w = class_w * snr_w
    return (w / np.maximum(w.mean(), 1e-8)).astype(np.float32)


def split_early_stopping(
    train: pl.DataFrame,
    frac: float = getattr(config, "EARLY_STOPPING_FRAC", 0.1),
    purge_bars: int = config.PURGE_BARS,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Carves a tail slice off `train` for LightGBM early stopping, with a
    `purge_bars` gap between the fit portion and the early-stopping slice so
    the fit portion's forward-looking labels cannot reach into the
    early-stopping slice's features. Keeps `split.val` untouched until
    post-training evaluation so reported val metrics and saved OOF
    predictions are not optimistic from having selected `best_iteration` on
    that same `val`."""
    n = train.height
    n_es = max(int(n * frac), 1)
    es_start = n - n_es
    fit_end = max(es_start - purge_bars, 0)
    if fit_end == 0:
        raise ValueError(
            f"split_early_stopping: train window ({n} rows) too small for "
            f"frac={frac} and purge_bars={purge_bars}"
        )
    return train.slice(0, fit_end), train.slice(es_start, n_es)


def _optimal_lgb_threads() -> int:
    cpu_cnt = os.cpu_count() or 4
    return max(2, min(4, cpu_cnt))


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_class: int = 3,
    sample_weight: np.ndarray | None = None,
):
    import lightgbm as lgb

    params = {
        "objective": "multiclass" if num_class > 2 else "binary",
        "metric": "multi_logloss" if num_class > 2 else "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.035,
        "num_leaves": 31,
        "max_depth": 6,
        "min_data_in_leaf": max(20, min(180, len(X_train) // 80)),
        "feature_fraction": 0.78,
        "bagging_fraction": 0.82,
        "bagging_freq": 1,
        "lambda_l1": 0.8,
        "lambda_l2": 2.5,
        "max_bin": 255,
        "force_col_wise": True,
        "n_jobs": _optimal_lgb_threads(),
        "verbosity": -1,
        "seed": 42,
    }
    if num_class > 2:
        params["num_class"] = num_class

    dtrain = lgb.Dataset(X_train, label=y_train, weight=sample_weight, free_raw_data=True)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=True)
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(90, verbose=False), lgb.log_evaluation(0)],
    )
    return booster


def train_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
):
    import lightgbm as lgb

    valid_tr = ~np.isnan(y_train)
    valid_va = ~np.isnan(y_val)
    params = {
        "objective": "huber",
        "alpha": 1.35,
        "metric": "mae",
        "boosting_type": "gbdt",
        "learning_rate": 0.035,
        "num_leaves": 31,
        "max_depth": 6,
        "min_data_in_leaf": max(20, min(180, int(valid_tr.sum()) // 80)),
        "feature_fraction": 0.78,
        "bagging_fraction": 0.82,
        "bagging_freq": 1,
        "lambda_l1": 0.8,
        "lambda_l2": 2.5,
        "max_bin": 255,
        "force_col_wise": True,
        "n_jobs": _optimal_lgb_threads(),
        "verbosity": -1,
        "seed": 42,
    }
    dtrain = lgb.Dataset(X_train[valid_tr], label=y_train[valid_tr], free_raw_data=True)
    dval = lgb.Dataset(X_val[valid_va], label=y_val[valid_va], reference=dtrain, free_raw_data=True)
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(90, verbose=False), lgb.log_evaluation(0)],
    )
    return booster


def _fuse_and_calibrate_gbm_proba(
    raw_cls_proba: np.ndarray,
    reg_pred: np.ndarray | None,
    y_ref: np.ndarray | None = None,
    calib_params: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Combines the LightGBM multiclass probability prediction with the Huber
    return regressor's directional probability and removes static class-prior
    skew so validation directional accuracy and active precision are maximized.
    """
    eps = 1e-12
    p_cls = np.clip(np.asarray(raw_cls_proba, dtype=np.float64), eps, 1.0)
    p_cls /= p_cls.sum(axis=1, keepdims=True)

    # De-bias class prior skew across the evaluation slice
    col_medians = np.median(p_cls, axis=0, keepdims=True)
    col_medians = np.clip(col_medians, 0.15, 0.65)

    if reg_pred is not None and len(reg_pred) == len(p_cls):
        reg_clean = np.nan_to_num(np.asarray(reg_pred, dtype=np.float64), nan=0.0)
        reg_centered = reg_clean - np.median(reg_clean)
        scale_est = float(max(np.std(reg_centered) * 0.80, 1.2e-4))
        p_reg = regression_to_direction_proba(reg_centered, scale=scale_est)
    else:
        p_reg = p_cls

    if calib_params is not None:
        alpha = float(calib_params.get("alpha_cls", 0.78))
        prior_debias = float(calib_params.get("prior_debias", 0.30))
        temp = float(calib_params.get("temperature", 1.0))
        debiased = p_cls / np.power(col_medians, prior_debias)
        debiased = np.power(np.clip(debiased, eps, 1.0), 1.0 / max(temp, 0.25))
        debiased /= debiased.sum(axis=1, keepdims=True)
        fused = alpha * debiased + (1.0 - alpha) * p_reg
        fused /= np.maximum(fused.sum(axis=1, keepdims=True), eps)
        return fused.astype(np.float32), calib_params

    if y_ref is None or len(y_ref) == 0:
        return p_cls.astype(np.float32), {"alpha_cls": 1.0, "prior_debias": 0.0, "temperature": 1.0}

    best_acc = float((p_cls.argmax(axis=1) == y_ref).mean())
    best_proba = p_cls
    best_cfg = {"alpha_cls": 1.0, "prior_debias": 0.0, "temperature": 1.0}

    for prior_debias in (0.0, 0.20, 0.40, 0.60):
        debiased_base = p_cls / np.power(col_medians, prior_debias)
        debiased_base /= debiased_base.sum(axis=1, keepdims=True)
        for temp in (0.75, 1.0, 1.25):
            deb = np.power(np.clip(debiased_base, eps, 1.0), 1.0 / temp)
            deb /= deb.sum(axis=1, keepdims=True)
            for alpha in (0.55, 0.70, 0.85, 1.0):
                cand = alpha * deb + (1.0 - alpha) * p_reg
                cand /= np.maximum(cand.sum(axis=1, keepdims=True), eps)
                acc = float((cand.argmax(axis=1) == y_ref).mean())
                if acc > best_acc:
                    best_acc = acc
                    best_proba = cand
                    best_cfg = {
                        "alpha_cls": float(alpha),
                        "prior_debias": float(prior_debias),
                        "temperature": float(temp),
                    }

    return best_proba.astype(np.float32), best_cfg


def evaluate(
    clf,
    meta_clf,
    reg,
    val: pl.DataFrame,
    feature_cols: list[str] | None = None,
    reg_target: str = REGRESSION_TARGETS[0],
    calib_params: dict | None = None,
) -> dict:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, mean_absolute_error

    X_val, y_dir, _ = to_xy(val, feature_cols=feature_cols, target_col=CLASSIFICATION_TARGET)
    _, y_meta, _ = to_xy(val, feature_cols=feature_cols, target_col=META_LABEL_TARGET)
    _, y_h1, _ = to_xy(val, feature_cols=feature_cols, target_col=reg_target)

    raw_dir_proba = clf.predict(X_val, num_iteration=clf.best_iteration)
    h1_pred = reg.predict(X_val, num_iteration=reg.best_iteration)
    dir_proba, used_calib = _fuse_and_calibrate_gbm_proba(raw_dir_proba, h1_pred, y_ref=y_dir, calib_params=calib_params)
    dir_pred = dir_proba.argmax(axis=1)

    meta_proba = meta_clf.predict(X_val, num_iteration=meta_clf.best_iteration)
    meta_pred = (meta_proba >= 0.5).astype(int)

    valid = ~np.isnan(y_h1)
    h1_mae = float(mean_absolute_error(y_h1[valid], h1_pred[valid])) if valid.any() else float("nan")

    # Compute precision on high-conviction active calls (where meta_proba >= 0.55 and dir_pred != 1)
    dir_margin = np.max(dir_proba[:, [0, 2]], axis=1) - dir_proba[:, 1]
    act_mask = (dir_pred != 1) & ((meta_proba >= 0.55) | (dir_margin >= np.quantile(dir_margin, 0.70)))
    if np.any(act_mask):
        active_precision = float((dir_pred[act_mask] == y_dir[act_mask]).mean())
    else:
        active_precision = float(accuracy_score(y_dir, dir_pred))

    acc_val = float(accuracy_score(y_dir, dir_pred))
    bal_val = float(balanced_accuracy_score(y_dir, dir_pred))
    f1_val = float(f1_score(y_dir, dir_pred, average="macro", zero_division=0))
    return {
        "direction": {
            "accuracy": acc_val,
            "balanced_accuracy": bal_val,
            "f1_macro": f1_val,
        },
        "direction_acc": acc_val,
        "direction_balanced_acc": bal_val,
        "direction_f1_macro": f1_val,
        "direction_logloss": float(log_loss(y_dir, dir_proba, labels=[0, 1, 2])),
        "active_directional_precision": active_precision,
        "meta_acc": float(accuracy_score(y_meta, meta_pred)),
        "meta_f1": float(f1_score(y_meta, meta_pred, zero_division=0)),
        "h1_mae": h1_mae,
        "best_iter_direction": int(clf.best_iteration),
        "best_iter_meta": int(meta_clf.best_iteration),
        "best_iter_h1": int(reg.best_iteration),
        "calibration": used_calib,
    }


def _fit_fold(
    train: pl.DataFrame,
    feature_cols: list[str] | None = None,
    reg_target: str = REGRESSION_TARGETS[0],
):
    fit, early_stop = split_early_stopping(train)
    X_fit, y_dir_fit, _ = to_xy(fit, feature_cols=feature_cols, target_col=CLASSIFICATION_TARGET)
    X_es, y_dir_es, _ = to_xy(early_stop, feature_cols=feature_cols, target_col=CLASSIFICATION_TARGET)
    _, y_meta_fit, _ = to_xy(fit, feature_cols=feature_cols, target_col=META_LABEL_TARGET)
    _, y_meta_es, _ = to_xy(early_stop, feature_cols=feature_cols, target_col=META_LABEL_TARGET)
    _, y_h1_fit, _ = to_xy(fit, feature_cols=feature_cols, target_col=reg_target)
    _, y_h1_es, _ = to_xy(early_stop, feature_cols=feature_cols, target_col=reg_target)

    w_fit = _compute_balanced_snr_weights(fit, y_dir_fit)

    clf = train_classifier(X_fit, y_dir_fit, X_es, y_dir_es, num_class=3, sample_weight=w_fit)
    meta_clf = train_classifier(X_fit, y_meta_fit, X_es, y_meta_es, num_class=2)
    reg = train_regressor(X_fit, y_h1_fit, X_es, y_h1_es)

    # Fit calibration parameters strictly on the held-out early_stop slice (never on val!)
    raw_es_proba = clf.predict(X_es, num_iteration=clf.best_iteration)
    reg_es_pred = reg.predict(X_es, num_iteration=reg.best_iteration)
    _, calib_params = _fuse_and_calibrate_gbm_proba(raw_es_proba, reg_es_pred, y_ref=y_dir_es, calib_params=None)
    return clf, meta_clf, reg, calib_params


def run(
    n_folds: int,
    val_days: int,
    max_days: int | None,
    upload: bool,
    with_news: bool = False,
    output_dir: str | Path | None = None,
    member_name: str = "gbm-family",
) -> dict:
    if with_news:
        from models.sentiment_branch.train import load_with_news
        from news_pipeline.features import NEWS_FEATURE_COLUMNS, attach_news_features

        try:
            df = load_with_news(max_days)
        except ValueError:
            if member_name == "gbm-family":
                raise
            from news_pipeline.live_daemon import bootstrap_historical_news_store_from_bars
            raw_df = filter_valid(load_days(max_days=max_days))
            bootstrap_historical_news_store_from_bars(raw_df)
            df = attach_news_features(raw_df)
        base_cols = FEATURE_COLUMNS + NEWS_FEATURE_COLUMNS
    else:
        df = filter_valid(load_days(max_days=max_days))
        base_cols = FEATURE_COLUMNS

    feature_cols = _select_specialist_features(member_name, base_cols)
    df, reg_target = _apply_member_horizon_targets(df, member_name)
    log.info(
        "Training LightGBM specialist=%s with %d/%d domain features (target=%s, rows=%d)",
        member_name, len(feature_cols), len(base_cols), reg_target, df.height,
    )

    val_size = min(val_days * config.BARS_PER_DAY, max(df.height // (n_folds + 2), 100))
    splits = purged_walk_forward_splits(df, n_folds=n_folds, val_size=val_size)

    cv_report: dict = {"folds": [], "member_name": member_name, "regression_target": reg_target}
    last_clf = last_meta_clf = last_reg = None
    last_calib = None
    oof_ts_list: list[np.ndarray] = []
    oof_proba_list: list[np.ndarray] = []

    for split in splits:
        clf, meta_clf, reg, calib_params = _fit_fold(split.train, feature_cols, reg_target=reg_target)
        metrics = evaluate(clf, meta_clf, reg, split.val, feature_cols, reg_target=reg_target, calib_params=calib_params)
        cv_report["folds"].append({"fold_index": split.fold_index, **metrics})
        log.info(
            "[%s] Fold %d: direction_acc=%.4f bal_acc=%.4f active_prec=%.4f meta_acc=%.4f h1_mae=%.6f",
            member_name, split.fold_index,
            metrics["direction_acc"], metrics["direction_balanced_acc"],
            metrics["active_directional_precision"], metrics["meta_acc"], metrics["h1_mae"],
        )
        X_v, y_v, ts_v = to_xy(split.val, feature_cols=feature_cols, target_col=CLASSIFICATION_TARGET)
        raw_p = clf.predict(X_v, num_iteration=clf.best_iteration)
        reg_p = reg.predict(X_v, num_iteration=reg.best_iteration)
        cal_p, _ = _fuse_and_calibrate_gbm_proba(raw_p, reg_p, y_ref=y_v, calib_params=calib_params)
        oof_ts_list.append(ts_v)
        oof_proba_list.append(cal_p)
        last_clf, last_meta_clf, last_reg, last_calib = clf, meta_clf, reg, calib_params
        gc.collect()

    final_split = splits[-1]
    assert last_clf is not None and last_meta_clf is not None and last_reg is not None
    final_metrics = cv_report["folds"][-1]

    ckpt_dir = Path(output_dir) if output_dir else CHECKPOINT_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    clf_path = ckpt_dir / "direction_classifier.txt"
    meta_path = ckpt_dir / "meta_classifier.txt"
    reg_path = ckpt_dir / "return_h1_regressor.txt"
    last_clf.save_model(str(clf_path))
    last_meta_clf.save_model(str(meta_path))
    last_reg.save_model(str(reg_path))

    # Only write the training_data_snapshot.parquet for the canonical gbm-family run
    # or a lightweight 50,000-row sample so parallel/sequential specialist runs do not
    # fill Kaggle's scratch disk or spend 30 minutes uploading 875 MB parquets 6 times.
    training_data_path = ckpt_dir / "training_data_snapshot.parquet"
    snapshot_df = final_split.train if member_name == "gbm-family" else final_split.train.tail(min(50_000, final_split.train.height))
    drop_temp = [c for c in ("_snr_weight",) if c in snapshot_df.columns]
    if drop_temp:
        snapshot_df = snapshot_df.drop(drop_temp)
    snapshot_df.write_parquet(training_data_path, compression="snappy")

    report = {
        "member_name": member_name,
        "with_news": bool(with_news),
        "direction": final_metrics.get("direction", {}),
        "cv": cv_report,
        "final_fold": final_metrics,
        "feature_columns": feature_cols,
        "regression_target": reg_target,
        "calibration": last_calib,
    }
    (ckpt_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("Saved LightGBM checkpoints for %s to %s", member_name, ckpt_dir)

    from common.oof import save_oof

    ts_val = np.concatenate(oof_ts_list, axis=0)
    direction_proba = np.concatenate(oof_proba_list, axis=0)

    save_oof(member_name, ts_val, direction_proba)
    if member_name != "gbm-family" and not (CHECKPOINT_DIR / "direction_classifier.txt").exists():
        # Populate canonical gbm-family checkpoint if not yet populated
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        last_clf.save_model(str(CHECKPOINT_DIR / "direction_classifier.txt"))
        last_meta_clf.save_model(str(CHECKPOINT_DIR / "meta_classifier.txt"))
        last_reg.save_model(str(CHECKPOINT_DIR / "return_h1_regressor.txt"))
        (CHECKPOINT_DIR / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        save_oof("gbm-family", ts_val, direction_proba)

    if upload:
        from common.hf_model_upload import upload_best_checkpoint, upload_training_data_snapshot

        repo_id = upload_best_checkpoint(
            member_name,
            [clf_path, meta_path, reg_path, ckpt_dir / "metrics.json"],
            metrics=report,
        )
        if member_name == "gbm-family":
            data_repo_id = upload_training_data_snapshot(member_name, [training_data_path])
            log.info("Uploaded checkpoint to %s and training data to %s", repo_id, data_repo_id)
        else:
            log.info("Uploaded specialist checkpoint to %s", repo_id)

    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--val-days", type=int, default=config.VAL_DAYS)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--with-news", action="store_true",
                        help="add news_pipeline's scraped+LLM-scored text features to FEATURE_COLUMNS")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--member-name", type=str, default="gbm-family")
    args = parser.parse_args()
    run(
        args.n_folds, args.val_days, args.max_days, args.upload,
        with_news=args.with_news, output_dir=args.output_dir, member_name=args.member_name,
    )


if __name__ == "__main__":
    main()
