"""
Kaggle-side training session: models.train_all under an orchestration.
checkpoint_manager.SessionBudget, with every member uploading to (and, on a
relaunch, restoring from) its private HF repo.

Not run directly as the kernel's code file — Kaggle only receives one file,
so orchestration/kaggle_relay/build_bundle.py packs the repo into a
self-contained kaggle_session_bundle.py whose bootstrap unpacks the code,
installs requirements-models.txt and then runs this module. That same
bundle is what poll_and_relaunch.py pushes, and what you upload/run by hand
on a dual-T4 Kaggle VM.

Per session:
  1. HF_TOKEN from the environment or Kaggle Secrets (add-on "Secrets",
     label HF_TOKEN) — required: data download, checkpoint upload/restore.
  2. News: TRAIN_WITH_NEWS=auto (default) pulls the scored store
     (news_pipeline.store_sync) and trains gbm-family + sentiment-branch
     with --with-news iff it produced articles; "1" requires it, "0" skips.
     NEWS_REFRESH_ON_KAGGLE=1 (Internet on) first tops the store up on this
     VM: incremental scrape + scoring on both T4s + push
     (news_pipeline.refresh); GPUs are released before training starts.
  3. train_all.run(upload=True, resume=True, budget=...) — finished members
     are restored from HF instead of retrained, GPU members auto-detect
     both T4s (one single-GPU retry on failure), and the fusion stages run
     once every member is finished or given up.

Usage (inside the unpacked code dir; the bundle does this for you):
    python -m orchestration.kaggle_relay.run_kaggle_session
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


def ensure_hf_token() -> None:
    if os.environ.get("HF_TOKEN"):
        return
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore

        os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as e:
        raise RuntimeError("HF_TOKEN is not set — add it as a Kaggle Secret labelled HF_TOKEN "
                           "(Add-ons -> Secrets) and attach it to this notebook") from e


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
    budget = SessionBudget(budget_hours=config.KAGGLE_SESSION_TIME_BUDGET_HOURS)
    ensure_hf_token()
    with_news = decide_with_news(already_pulled=refresh_news())
    log.info("Kaggle session starting: %.1fh budget (safety margin %.0fs), with_news=%s, %d GPU(s)",
              config.KAGGLE_SESSION_TIME_BUDGET_HOURS, budget.safety_margin_seconds, with_news, train_all._cuda_count())
    skip = [s for s in os.environ.get("TRAIN_SKIP", "").split(",") if s]
    resume = os.environ.get("TRAIN_RESUME", "1").lower() not in ("0", "false", "no")
    force_retrain_env = os.environ.get("FORCE_RETRAIN", "")
    force_retrain = [s.strip() for s in force_retrain_env.split(",") if s.strip()]
    log.info("Session training flags: resume=%s, skip=%s, force_retrain=%s", resume, skip, force_retrain)
    # Ensure contracts and interfaces are registered in sys.modules
    try:
        import contracts_interface  # registers tests.contracts alias
    except Exception as e:
        log.warning("Could not register contracts_interface: %s", e)

    # Ensure training data is safely downloaded or bootstrapped once before spawning parallel lanes
    try:
        from common.data import ensure_training_data_ready
        ensure_training_data_ready(symbol=config.SYMBOL_UM, min_days=42)
    except Exception as e:
        log.warning("Data pre-initialization warning: %s", e)

    # Launch Project-IMPJR v4.0 Multi-Horizon Hierarchical Training Wave
    try:
        from models_v3 import train_all_multi_horizon
        log.info("Launching Project-IMPJR v4.0 Multi-Horizon Training Wave across all 4 tiers...")
        v4_results = train_all_multi_horizon.run(budget=budget, force_retrain=force_retrain)
        log.info("v4.0 Multi-Horizon Training Wave status: %s", v4_results)
    except Exception as e:
        log.exception("v4.0 Multi-Horizon Training encountered error: %s", e)

    results = train_all.run(
        val_days=config.VAL_DAYS, max_days=None, upload=True, skip=skip,
        members_only=False, gpus=None, resume=resume, budget=budget, with_news=with_news,
        force_retrain=force_retrain,
    )
    log.info("Session training verification finished after %.2fh: %s", budget.elapsed_seconds() / 3600, results)
    failed = [k for k, v in results.items() if v in {"failed", "timeout"}]
    if failed:
        log.error("Failed stages this session: %s", failed)
    
    # Launch Live Streaming Prediction Engine with Dual T4 GPUs on Kaggle
    log.info("=" * 70)
    log.info("STARTING LIVE PREDICTION & STREAMING ENGINE ON KAGGLE DUAL-T4 VM")
    log.info("=" * 70)
    try:
        from streaming import live_stream_glue
        live_stream_glue.main()
    except Exception:
        log.exception("Live streaming loop exited")


if __name__ == "__main__":
    main()
