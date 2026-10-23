"""
Sentiment-fusion branch ensemble member.

Two input modes:

- default: a classifier over the already-computed features that most
  directly proxy crowd positioning/sentiment — funding rate, long/short
  ratios, taker long/short volume ratio, funding-OI crowding, and implied
  vol (SENTIMENT_PROXY_FEATURES). Needs no text data.
- --with-news: the same proxies PLUS the real text-sentiment features from
  news_pipeline (scraped full-text articles scored by the two-tier
  CryptoBERT + LLM scorer, aggregated point-in-time over 15m/1h/4h/24h —
  see news_pipeline/features.py). Requires a scored news store covering the
  training window (news_pipeline.scraper backfill + news_pipeline.llm_scorer
  --once); refuses to train if the store covers none of it.

The trained column list is written to metrics.json["feature_columns"];
inference/checkpoint_loaders.py reads it and attaches the news features at
inference time only when the checkpoint was trained with them.

Ablation mode (§9.3 sentiment ablation): --ablation trains a SEPARATE pair
of full-feature-set LightGBM classifiers — one including
SENTIMENT_PROXY_FEATURES, one with them dropped — on the exact same
train/val split, and exports each side's OOF as "sentiment_branch_with" /
"sentiment_branch_without" so validation/ablation_test.py's own documented
example usage becomes runnable:

    python -m validation.ablation_test --variant sentiment_branch_with --baseline sentiment_branch_without

This is deliberately a different comparison than the module's default
run(): the default trains a small classifier over ONLY the 13 proxy
features (feeding live_loop.py's "sentiment-branch" member); the ablation
pair instead asks "does adding these 13 columns to the FULL feature set
change anything," which is what an ablation test is actually meant to
measure. --ablation writes to its own checkpoints/ablation/ subdirectory
and never touches sentiment_proxy_classifier.txt, so it can't interfere
with the live-loop-consumed default checkpoint.

--ablation --with-news instead compares FEATURE_COLUMNS + NEWS_FEATURE_COLUMNS
against FEATURE_COLUMNS alone — the §9.3 "does text sentiment earn its
place" test, judged by validation.ablation_test against the §10.3 kill
criterion (>= 1% relative improvement or it's dropped).

Usage:
    python -m models.sentiment_branch.train --val-days 7 --upload
    python -m models.sentiment_branch.train --with-news --val-days 7 --upload
    python -m models.sentiment_branch.train --ablation --val-days 7
    python -m models.sentiment_branch.train --ablation --with-news --val-days 7
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import polars as pl

import config
from common.cv import purged_walk_forward_splits
from common.data import filter_valid, load_days, to_xy
from common.feature_contract import CLASSIFICATION_TARGET, FEATURE_COLUMNS

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


def _fit_and_early_stop_sets(train: pl.DataFrame, feature_cols: list[str], target: str = CLASSIFICATION_TARGET):
    """(fit, early-stop) lgb.Datasets from the training window only (see
    models.gbm_family.train.split_early_stopping) — the val fold stays
    untouched until scoring, so with/without comparisons aren't each tuned
    on the rows they're compared on."""
    import lightgbm as lgb

    from models.gbm_family.train import split_early_stopping

    fit, early_stop = split_early_stopping(train)
    X_fit, y_fit, _ = to_xy(fit, feature_cols=feature_cols, target_col=target)
    X_es, y_es, _ = to_xy(early_stop, feature_cols=feature_cols, target_col=target)
    return (lgb.Dataset(X_fit, label=y_fit, free_raw_data=False),
            lgb.Dataset(X_es, label=y_es, free_raw_data=False))


def train_and_eval(train: pl.DataFrame, val: pl.DataFrame, feature_cols: list[str] | None = None, target: str = CLASSIFICATION_TARGET):
    import lightgbm as lgb
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    feature_cols = feature_cols or SENTIMENT_PROXY_FEATURES
    train_set, val_set = _fit_and_early_stop_sets(train, feature_cols, target=target)
    X_val, y_val, _ = to_xy(val, feature_cols=feature_cols, target_col=target)
    params = {
        "objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
        "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 200, "verbosity": -1,
    }
    booster = lgb.train(params, train_set, num_boost_round=1000, valid_sets=[val_set],
                         callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])

    pred = booster.predict(X_val, num_iteration=booster.best_iteration).argmax(axis=1)
    metrics = {
        "accuracy": float(accuracy_score(y_val, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_val, pred)),
        "f1_macro": float(f1_score(y_val, pred, average="macro", zero_division=0)),
        "best_iteration": int(booster.best_iteration),
        "is_placeholder_proxy": feature_cols == SENTIMENT_PROXY_FEATURES,
        "proxy_features": SENTIMENT_PROXY_FEATURES,
        "feature_columns": feature_cols,
    }
    return booster, metrics


def run(
    val_days: int,
    max_days: int | None,
    upload: bool,
    with_news: bool = False,
    member_name: str = MEMBER_NAME,
    output_dir: Path | str | None = None,
) -> dict:
    f1_leadlag = [
        "eth_lead_return_lag_1", "eth_lead_return_lag_2", "eth_lead_return_lag_3",
        "cross_return_spread_10s", "cross_return_spread_30s", "cross_return_residual_10s",
        "cross_ofi_divergence_10s", "cross_ofi_divergence_30s", "cross_mom_diff_30s",
        "cross_mom_diff_zscore_30s", "eth_lead_ret_bps", "eth_imbalance",
    ]
    target_col = CLASSIFICATION_TARGET
    if with_news or member_name == "tier2-cryptobert-news":
        from news_pipeline.features import NEWS_FEATURE_COLUMNS

        df = load_with_news(max_days)
        feature_cols = SENTIMENT_PROXY_FEATURES + NEWS_FEATURE_COLUMNS
    else:
        df = filter_valid(load_days(max_days=max_days))
        if member_name == "tier2-crossasset-gat":
            available_leadlag = [c for c in f1_leadlag if c in df.columns]
            feature_cols = SENTIMENT_PROXY_FEATURES + available_leadlag
        else:
            feature_cols = SENTIMENT_PROXY_FEATURES

    if member_name in ("tier2-crossasset-gat", "tier2-cryptobert-news"):
        if "close" in df.columns and "high" in df.columns and "low" in df.columns:
            try:
                from labeling_multi_horizon_v4 import fee_floored_triple_barrier_labels_v4
                df = fee_floored_triple_barrier_labels_v4(
                    df, max_hold_bars=6, pt_mult=1.5, sl_mult=1.0, min_pt_bps=14.0, min_sl_bps=10.0,
                    maker_tp_fee_bps=4.0, passive_unwind_fee_bps=5.0, taker_sl_fee_bps=7.5, prefix="tactical_5m",
                )
                target_col = "label_tactical_5m_tb"
            except Exception as exc:
                log.warning("Could not apply tactical_5m triple barrier: %s", exc)

    val_size = val_days * config.BARS_PER_DAY
    split = purged_walk_forward_splits(df, n_folds=1, val_size=val_size)[0]

    booster, metrics = train_and_eval(split.train, split.val, feature_cols, target=target_col)

    target_chk_dir = Path(output_dir) if output_dir else CHECKPOINT_DIR
    target_chk_dir.mkdir(parents=True, exist_ok=True)
    model_path = target_chk_dir / "sentiment_proxy_classifier.txt"
    booster.save_model(str(model_path))

    training_data_path = target_chk_dir / "training_data_snapshot.parquet"
    split.train.select(["ts", *feature_cols, target_col]).write_parquet(
        training_data_path, compression="snappy"
    )
    (target_chk_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log.info("[%s] Sentiment branch (%s): accuracy=%.4f (target=%s)", member_name, "proxies + news" if with_news else "proxies only", metrics["accuracy"], target_col)

    from common.oof import save_oof

    X_val, _, ts_val = to_xy(split.val, feature_cols=feature_cols, target_col=target_col)
    direction_proba = booster.predict(X_val, num_iteration=booster.best_iteration)
    save_oof(member_name, ts_val, direction_proba)
    if member_name == MEMBER_NAME:
        save_oof(MEMBER_NAME, ts_val, direction_proba)

    if upload:
        from common.hf_model_upload import upload_best_checkpoint, upload_training_data_snapshot

        repo_id = upload_best_checkpoint(MEMBER_NAME, [model_path, CHECKPOINT_DIR / "metrics.json"], metrics=metrics)
        data_repo_id = upload_training_data_snapshot(MEMBER_NAME, [training_data_path])
        log.info("Uploaded checkpoint to %s and training data to %s", repo_id, data_repo_id)

    return metrics


def _train_one_side(train: pl.DataFrame, val: pl.DataFrame, feature_cols: list[str]):
    import lightgbm as lgb
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    train_set, val_set = _fit_and_early_stop_sets(train, feature_cols)
    X_val, y_val, ts_val = to_xy(val, feature_cols=feature_cols, target_col=CLASSIFICATION_TARGET)
    params = {
        "objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
        "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 200, "verbosity": -1,
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
    args = parser.parse_args()
    if args.ablation:
        run_ablation(args.val_days, args.max_days, with_news=args.with_news)
    else:
        run(args.val_days, args.max_days, args.upload, with_news=args.with_news, member_name=args.member_name, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
