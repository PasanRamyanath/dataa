import re
from pathlib import Path

train_path = Path("feature engineering scripts/models/sentiment_branch/train.py")
text = train_path.read_text(encoding="utf-8")

old_fit = "def _fit_and_early_stop_sets(train: pl.DataFrame, feature_cols: list[str]):"
new_fit = "def _fit_and_early_stop_sets(train: pl.DataFrame, feature_cols: list[str], target: str = CLASSIFICATION_TARGET):"
text = text.replace(old_fit, new_fit, 1)

text = text.replace(
    "X_fit, y_fit, _ = to_xy(fit, feature_cols=feature_cols, target_col=CLASSIFICATION_TARGET)",
    "X_fit, y_fit, _ = to_xy(fit, feature_cols=feature_cols, target_col=target)",
    1,
)
text = text.replace(
    "X_es, y_es, _ = to_xy(early_stop, feature_cols=feature_cols, target_col=CLASSIFICATION_TARGET)",
    "X_es, y_es, _ = to_xy(early_stop, feature_cols=feature_cols, target_col=target)",
    1,
)

old_eval = "def train_and_eval(train: pl.DataFrame, val: pl.DataFrame, feature_cols: list[str] | None = None):"
new_eval = "def train_and_eval(train: pl.DataFrame, val: pl.DataFrame, feature_cols: list[str] | None = None, target: str = CLASSIFICATION_TARGET):"
text = text.replace(old_eval, new_eval, 1)

text = text.replace(
    "train_set, val_set = _fit_and_early_stop_sets(train, feature_cols)",
    "train_set, val_set = _fit_and_early_stop_sets(train, feature_cols, target=target)",
    1,
)
text = text.replace(
    "X_val, y_val, _ = to_xy(val, feature_cols=feature_cols, target_col=CLASSIFICATION_TARGET)",
    "X_val, y_val, _ = to_xy(val, feature_cols=feature_cols, target_col=target)",
    1,
)

old_run_sig = "def run(val_days: int, max_days: int | None, upload: bool, with_news: bool = False) -> dict:"
new_run_sig = """def run(
    val_days: int,
    max_days: int | None,
    upload: bool,
    with_news: bool = False,
    member_name: str = MEMBER_NAME,
    output_dir: Path | str | None = None,
) -> dict:"""
text = text.replace(old_run_sig, new_run_sig, 1)

# Update run body
old_run_body = """    if with_news:
        from news_pipeline.features import NEWS_FEATURE_COLUMNS

        df = load_with_news(max_days)
        feature_cols = SENTIMENT_PROXY_FEATURES + NEWS_FEATURE_COLUMNS
    else:
        df = filter_valid(load_days(max_days=max_days))
        feature_cols = SENTIMENT_PROXY_FEATURES
    val_size = val_days * config.BARS_PER_DAY
    split = purged_walk_forward_splits(df, n_folds=1, val_size=val_size)[0]

    booster, metrics = train_and_eval(split.train, split.val, feature_cols)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = CHECKPOINT_DIR / "sentiment_proxy_classifier.txt"
    booster.save_model(str(model_path))

    training_data_path = CHECKPOINT_DIR / "training_data_snapshot.parquet"
    split.train.select(["ts", *feature_cols, "label_triple_barrier"]).write_parquet(
        training_data_path, compression="snappy"
    )
    (CHECKPOINT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log.info("Sentiment branch (%s): accuracy=%.4f", "proxies + news" if with_news else "proxies only", metrics["accuracy"])

    from common.oof import save_oof

    X_val, _, ts_val = to_xy(split.val, feature_cols=feature_cols, target_col=CLASSIFICATION_TARGET)
    direction_proba = booster.predict(X_val, num_iteration=booster.best_iteration)
    save_oof(MEMBER_NAME, ts_val, direction_proba)"""

new_run_body = """    f1_leadlag = [
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
        save_oof(MEMBER_NAME, ts_val, direction_proba)"""

# Handle CRLF in text
text_normalized = text.replace("\r\n", "\n")
old_body_norm = old_run_body.replace("\r\n", "\n")
new_body_norm = new_run_body.replace("\r\n", "\n")
text_normalized = text_normalized.replace(old_body_norm, new_body_norm, 1)

old_args = """    parser.add_argument("--with-news", action="store_true",
                        help="add news_pipeline's scraped+LLM-scored text features (and, with --ablation, ablate them)")
    args = parser.parse_args()"""

new_args = """    parser.add_argument("--with-news", action="store_true",
                        help="add news_pipeline's scraped+LLM-scored text features (and, with --ablation, ablate them)")
    parser.add_argument("--member-name", type=str, default=MEMBER_NAME)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()"""

text_normalized = text_normalized.replace(old_args, new_args, 1)

old_run_call = "        run(args.val_days, args.max_days, args.upload, with_news=args.with_news)"
new_run_call = "        run(args.val_days, args.max_days, args.upload, with_news=args.with_news, member_name=args.member_name, output_dir=args.output_dir)"
text_normalized = text_normalized.replace(old_run_call, new_run_call, 1)

train_path.write_text(text_normalized, encoding="utf-8")
print("Successfully modified sentiment_branch/train.py")
