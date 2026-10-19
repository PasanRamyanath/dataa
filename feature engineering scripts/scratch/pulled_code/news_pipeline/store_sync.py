"""
Ships the scored news store (NEWS_ARTICLES_ROOT/scored/scored_*.jsonl — all
the training path reads) between the machine that scrapes/scores and the
Kaggle training VM, through a private HF dataset (config.HF_NEWS_REPO_ID).
Raw article bodies stay local; only the scored rows travel.

    python -m news_pipeline.store_sync push     # after scraper backfill + llm_scorer --once
    python -m news_pipeline.store_sync pull     # on Kaggle (run_kaggle_session does this)
"""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("news_pipeline.store_sync")

REPO_PREFIX = "scored"


def scored_dir() -> Path:
    return Path(config.NEWS_ARTICLES_ROOT) / "scored"


def push(api=None) -> int:
    from common.hf_model_upload import ensure_private_repo, get_api

    files = sorted(scored_dir().glob("scored_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No scored_*.jsonl under {scored_dir()} — run the scraper and "
                                "`python -m news_pipeline.llm_scorer --once` first")
    api = api or get_api()
    ensure_private_repo(config.HF_NEWS_REPO_ID, "dataset", api)
    api.upload_folder(folder_path=str(scored_dir()), path_in_repo=REPO_PREFIX, repo_id=config.HF_NEWS_REPO_ID,
                      repo_type="dataset", allow_patterns=["scored_*.jsonl"],
                      commit_message=f"Sync {len(files)} scored news file(s)")
    log.info("Pushed %d scored file(s) to %s", len(files), config.HF_NEWS_REPO_ID)
    return len(files)


def pull(api=None) -> int:
    """Returns the number of scored files now present locally; 0 (never
    raises) when the repo is missing or unreachable."""
    from huggingface_hub import hf_hub_download

    from common.hf_model_upload import get_api

    try:
        api = api or get_api()
        names = [f for f in api.list_repo_files(repo_id=config.HF_NEWS_REPO_ID, repo_type="dataset")
                 if f.startswith(f"{REPO_PREFIX}/scored_") and f.endswith(".jsonl")]
    except Exception as e:
        log.info("No news store on HF (%s): %s", config.HF_NEWS_REPO_ID, e)
        return 0
    out = scored_dir()
    out.mkdir(parents=True, exist_ok=True)
    pulled = 0
    for name in names:
        try:
            cached = hf_hub_download(repo_id=config.HF_NEWS_REPO_ID, filename=name, repo_type="dataset",
                                     token=config.get_hf_token())
            shutil.copyfile(cached, out / Path(name).name)
            pulled += 1
        except Exception as e:  # one bad file must not abort the training session
            log.warning("Could not pull %s: %s", name, e)
    log.info("Pulled %d/%d scored file(s) into %s", pulled, len(names), out)
    return pulled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["push", "pull"])
    args = parser.parse_args()
    push() if args.action == "push" else pull()


if __name__ == "__main__":
    main()
