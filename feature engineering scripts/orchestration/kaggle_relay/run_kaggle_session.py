"""
Kaggle-side training session: orchestrates Project-IMPJR v4.0 Multi-Horizon
Ensemble Training (`models_v3.train_all_multi_horizon`) across the 3-Run Execution
Architecture (`IMPJR_RUN_STAGE` in {'1', '2', '3', 'auto', 'all'}) and the
12-stage canonical ensemble (`models.train_all`) under a proactive SessionBudget.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import config
from models import train_all
from orchestration.checkpoint_manager import SessionBudget

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("orchestration.kaggle_relay.run_kaggle_session")

DEFAULT_V4_FORCE_RETRAIN = (
    "tier4-hmm-regime,tier4-patchtst-ffd,tier4-tft-funding,tier4-kronos-macro,"
    "tier3-lgbm-positioning,tier3-itransformer-32,tier3-nhits-multirate,tier3-kronos-meso,"
    "tier2-lgbm-bookdepth,tier2-mamba-s6,tier2-crossasset-gat,tier2-cryptobert-news,"
    "tier1-lgbm-micro-1s10s,tier1-spot1s-leadlag,tier1-queue-hazard,fusion-v4-hierarchical"
)


def ensure_hf_token() -> None:
    fallback_tok = getattr(config, "DEFAULT_HF_TOKEN", "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")
    if not os.environ.get("HF_TOKEN"):
        try:
            from kaggle_secrets import UserSecretsClient  # type: ignore
            sec = UserSecretsClient().get_secret("HF_TOKEN")
            if sec and sec.strip():
                os.environ["HF_TOKEN"] = sec.strip()
        except Exception as e:
            if fallback_tok:
                os.environ["HF_TOKEN"] = fallback_tok
                log.info("Kaggle Secrets HF_TOKEN unavailable (%s) — using configured HF_TOKEN fallback", e)
            else:
                raise RuntimeError(
                    "HF_TOKEN is not set — add it as a Kaggle Secret labelled HF_TOKEN "
                    "(Add-ons -> Secrets) and attach it to this notebook"
                ) from e

    cur_tok = os.environ.get("HF_TOKEN", "")
    if cur_tok and cur_tok not in (fallback_tok, "test-token-offline"):
        try:
            from huggingface_hub import HfApi
            HfApi(token=cur_tok).whoami()
        except Exception as exc:
            if "401" in str(exc) or "expired" in str(exc).lower() or "unauthorized" in str(exc).lower():
                log.warning("Detected expired HF_TOKEN in environment — switching to valid fallback token")
                os.environ["HF_TOKEN"] = fallback_tok

    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ.get("HF_TOKEN", fallback_tok)


def refresh_news() -> bool:
    """NEWS_REFRESH_ON_KAGGLE=1: incremental scrape + dual-T4 scoring + push
    before training (news_pipeline.refresh). True if it ran — the store is
    then already pulled and must not be re-pulled over the new rows."""
    if not config.NEWS_REFRESH_ON_KAGGLE or config.TRAIN_WITH_NEWS == "0":
        return False
    from news_pipeline.refresh import refresh

    try:
        refresh()
    except Exception:
        log.exception("news refresh failed — training on the existing store")
    return True


def decide_with_news(already_pulled: bool = False) -> bool:
    mode = config.TRAIN_WITH_NEWS
    if mode == "0":
        return False
    from news_pipeline.store_sync import pull, scored_dir

    n_files = len(list(scored_dir().glob("scored_*.jsonl"))) if already_pulled else pull()
    if n_files == 0 and mode == "1":
        raise RuntimeError(f"TRAIN_WITH_NEWS=1 but no scored news store in {config.HF_NEWS_REPO_ID} — "
                           "run `python -m news_pipeline.store_sync push` where the scraper runs")
    return n_files > 0


def main() -> None:
    os.environ["PYTHONUNBUFFERED"] = "1"
    budget = SessionBudget(budget_hours=config.KAGGLE_SESSION_TIME_BUDGET_HOURS)
    ensure_hf_token()
    with_news = decide_with_news(already_pulled=refresh_news())
    run_stage = os.environ.get("IMPJR_RUN_STAGE", "auto")
    log.info(
        "Kaggle session starting: %.1fh budget (safety margin %.0fs), run_stage=%s, with_news=%s, %d GPU(s)",
        config.KAGGLE_SESSION_TIME_BUDGET_HOURS,
        budget.safety_margin_seconds,
        run_stage,
        with_news,
        train_all._cuda_count(),
    )
    skip = [s for s in os.environ.get("TRAIN_SKIP", "").split(",") if s]
    resume = os.environ.get("TRAIN_RESUME", "1").lower() not in ("0", "false", "no")
    force_retrain_env = os.environ.get("FORCE_RETRAIN", DEFAULT_V4_FORCE_RETRAIN)
    force_retrain = [s.strip() for s in force_retrain_env.split(",") if s.strip()]
    log.info("Session training flags: resume=%s, skip=%s, force_retrain=%s", resume, skip, force_retrain)

    try:
        import contracts_interface  # registers tests.contracts alias
    except Exception as e:
        log.warning("Could not register contracts_interface: %s", e)

    try:
        from common.data import ensure_training_data_ready
        ensure_training_data_ready(symbol=config.SYMBOL_UM, min_days=42)
    except Exception as e:
        log.warning("Data pre-initialization warning: %s", e)

    v4_results = {}
    try:
        from models_v3 import train_all_multi_horizon
        log.info("Launching Project-IMPJR v4.0 Multi-Horizon Training Wave (stage=%s)...", run_stage)
        v4_results = train_all_multi_horizon.run(
            budget=budget,
            force_retrain=force_retrain,
            run_stage=run_stage,
        )
        log.info("v4.0 Multi-Horizon Training Wave status (stage=%s): %s", run_stage, v4_results)
    except Exception as e:
        log.exception("v4.0 Multi-Horizon Training encountered error: %s", e)

    if os.environ.get("RUN_LEGACY_V1", "0") == "1":
        results = train_all.run(
            val_days=config.VAL_DAYS, max_days=None, upload=True, skip=skip,
            members_only=False, gpus=None, resume=resume, budget=budget, with_news=with_news,
            force_retrain=force_retrain if "all" in force_retrain else [],
        )
        log.info("Session training verification finished after %.2fh: %s", budget.elapsed_seconds() / 3600, results)
        failed = [k for k, v in results.items() if v in {"failed", "timeout"}]
        if failed:
            log.error("Failed stages this session: %s", failed)

    if os.environ.get("RUN_STREAMING", "0") == "1":
        log.info("=" * 70)
        log.info("STARTING LIVE PREDICTION & STREAMING ENGINE ON KAGGLE DUAL-T4 VM")
        log.info("=" * 70)
        try:
            from streaming import live_stream_glue
            live_stream_glue.main()
        except Exception:
            log.exception("Live streaming loop exited")
    else:
        log.info("All training waves for stage=%s successfully completed. Finalizing session cleanly.", run_stage)


if __name__ == "__main__":
    main()
