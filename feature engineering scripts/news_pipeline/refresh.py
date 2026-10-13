"""
One-shot incremental news refresh, built for the dual-T4 Kaggle session
(orchestration/kaggle_relay/run_kaggle_session.py runs it before training
when config.NEWS_REFRESH_ON_KAGGLE=1; the notebook's Internet toggle must be
on). Also usable anywhere else as `python -m news_pipeline.refresh`.

  1. pull the scored store from config.HF_NEWS_REPO_ID
  2. scrape [max(newest scored article - 1 day, now - NEWS_REFRESH_MAX_DAYS), now):
     GDELT DOC + every outlet's sitemaps, domain-parallel and polite
     (news_pipeline.scraper). Already-scored article ids are pre-seeded as
     seen, so the 1-day overlap costs no refetches of known articles.
  3. score the new articles on every visible GPU (llm_scorer data-parallel
     replicas: vLLM if importable, else HF) and free the GPUs again
  4. push the store back (skipped when nothing new was scored)

Raw article bodies stay on the VM (only scored rows travel), exactly as
with the local scraper. Every step is best-effort: a failure is logged and
the session trains on whatever store it has.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("news_pipeline.refresh")


def newest_scored_ts(scored_root: Path) -> datetime | None:
    newest = None
    for path in sorted(Path(scored_root).glob("scored_*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    ts = json.loads(line).get("available_ts")
                except json.JSONDecodeError:
                    continue
                if ts and (newest is None or ts > newest):
                    newest = ts
    if newest is None:
        return None
    dt = datetime.fromisoformat(newest.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def refresh_window(scored_root: Path, now: datetime, max_days: int = config.NEWS_REFRESH_MAX_DAYS) -> tuple[datetime, datetime]:
    floor = now - timedelta(days=max_days)
    newest = newest_scored_ts(scored_root)
    start = floor if newest is None else max(floor, newest - timedelta(days=1))
    return start, now


def refresh(now: datetime | None = None, max_days: int = config.NEWS_REFRESH_MAX_DAYS, *,
            pull_fn=None, push_fn=None, scorer_factory=None, gdelt_fn=None, backfill_fn=None,
            fetcher=None) -> dict:
    """Returns a summary dict; never raises for a network/scoring failure
    (logged) so the caller's training still runs."""
    from news_pipeline import llm_scorer, scraper, store_sync

    now = now or datetime.now(timezone.utc)
    pull_fn = pull_fn or store_sync.pull
    push_fn = push_fn or store_sync.push
    scorer_factory = scorer_factory or llm_scorer.TwoTierScorer.from_config
    gdelt_fn = gdelt_fn or scraper.gdelt_backfill
    backfill_fn = backfill_fn or scraper.backfill
    scored_root = store_sync.scored_dir()
    summary = {"pulled": 0, "gdelt": 0, "sitemaps": 0, "scored": 0, "pushed": 0}

    try:
        summary["pulled"] = pull_fn()
    except Exception:
        log.exception("news refresh: pull failed — continuing from the local store")
    start, end = refresh_window(scored_root, now, max_days)
    summary["window"] = [start.isoformat(), end.isoformat()]
    log.info("news refresh window %s -> %s", start, end)

    store = scraper.ArticleStore()
    store.seen_ids().update(llm_scorer._scored_ids(scored_root))  # don't refetch what's already scored
    fetcher = fetcher or scraper.PoliteFetcher()
    for key, fn in (("gdelt", gdelt_fn), ("sitemaps", backfill_fn)):
        try:
            summary[key] = fn(start, end, fetcher=fetcher, store=store)
        except Exception:
            log.exception("news refresh: %s scrape failed", key)

    scorer = None
    try:
        if llm_scorer.pending_articles(store.root, scored_root):
            scorer = scorer_factory()
            summary["scored"] = llm_scorer.score_pending(scorer, raw_root=store.root, scored_root=scored_root)
    except Exception:
        log.exception("news refresh: scoring failed")
    finally:
        if scorer is not None:
            scorer.close()  # hand the GPUs back to training
            del scorer

    if summary["scored"]:
        try:
            summary["pushed"] = push_fn()
        except Exception:
            log.exception("news refresh: push failed — new scores exist on this VM only")
    log.info("news refresh done: %s", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-days", type=int, default=config.NEWS_REFRESH_MAX_DAYS)
    args = parser.parse_args()
    refresh(max_days=args.max_days)


if __name__ == "__main__":
    main()
