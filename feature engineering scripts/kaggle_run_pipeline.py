"""
Entry point for a Kaggle CPU notebook's "Save & Run All" (commit) execution.

Each commit run starts from a completely fresh VM and disk — nothing written
to /kaggle/working or anywhere else on the Kaggle filesystem by a previous
run is visible to this one. So this script never trusts any local file as
its record of "what's already done"; instead, at startup it asks the
HuggingFace dataset repo itself which days already have an uploaded
features_{symbol}_{date}.parquet, via a single list_repo_files() call, and
skips exactly those days. Each day's upload is one atomic upload_file() call
(see publish_to_hf.upload_daily_features), so a day is either fully present
in the repo or not there at all — there is no partial/corrupted per-day
state to reconcile, and re-running this script after a kill (Kaggle's ~12h
ceiling, a lost connection, anything) is always safe: it just re-derives the
remaining day list and keeps going.

For each remaining day, in order:
  1. download just that day's raw files (kaggle_downloader.ensure_day_files)
  2. build_dataset.build_day(day) — the existing, already-tested pipeline
  3. publish_to_hf.upload_daily_features(...)
  4. kaggle_downloader.cleanup_day_files(day) — delete the raw files again

Step 4 keeps peak temp-disk usage bounded to roughly one day's worth of raw
files regardless of how many days a single run processes, respecting
Kaggle's temp storage cap. An internal wall-clock budget (--time-budget-hours,
default 11 — under Kaggle's 12h hard ceiling) stops the day loop between
days, never mid-day, so a day is never left half-downloaded or half-built
when the process exits.

Usage (inside a Kaggle notebook cell, with HF_TOKEN set via Kaggle Secrets):
    python kaggle_run_pipeline.py --start 2023-01-01 --end 2026-09-11

Point the BINANCE_DATA_ROOT / BYBIT_TRADING_ROOT / etc. env vars (see
config.py) at scratch paths under /kaggle/temp (or wherever Kaggle exposes
writable temp space) before running, so downloaded raw files never land
under /kaggle/working — that path is capped at 20GB and becomes the
notebook's permanent Output, and this pipeline has no use for the raw
per-day files once a day is uploaded to HF.
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import date, timedelta

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

import config
import kaggle_downloader
from build_dataset import build_day
from publish_to_hf import get_api, upload_daily_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("kaggle_run_pipeline")

DEFAULT_TIME_BUDGET_HOURS = 11.0  # safety margin under Kaggle's ~12h hard ceiling
HF_CALL_MAX_RETRIES = 10
HF_CALL_BACKOFF_SECONDS = 5.0
INTER_DAY_PACING_SECONDS = 1.0  # small courtesy delay between per-day HF uploads


def _daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def _with_hf_retry(fn, *args, **kwargs):
    """Runs one HF Hub API call with exponential backoff, handling transient
    network errors and HTTP 429 rate limits gracefully."""
    last_exc = None
    for attempt in range(HF_CALL_MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except (HfHubHTTPError, OSError) as e:
            last_exc = e
            if attempt == HF_CALL_MAX_RETRIES - 1:
                break
            is_429 = "429" in str(e) or (hasattr(e, "response") and getattr(e.response, "status_code", None) == 429)
            if is_429:
                sleep_s = 65.0
                log.warning("HF API rate limit 429 (attempt %d/%d) — sleeping 65s before retrying...",
                            attempt + 1, HF_CALL_MAX_RETRIES)
            else:
                sleep_s = HF_CALL_BACKOFF_SECONDS * (2 ** attempt)
                log.warning("HF API call failed (attempt %d/%d): %s — retrying in %.0fs",
                            attempt + 1, HF_CALL_MAX_RETRIES, e, sleep_s)
            time.sleep(sleep_s)
    raise last_exc


def completed_dates(api: HfApi, symbol: str) -> set[str]:
    """Single list_repo_files() call to discover which YYYY-MM-DD days
    already have an uploaded parquet — the entire resumability mechanism.
    Deliberately not called per-day: one call per run keeps this well
    within HF's API rate limits regardless of how many days remain."""
    prefix = f"{config.HF_DATA_PREFIX}/features_{symbol}_"
    suffix = ".parquet"
    try:
        files = _with_hf_retry(api.list_repo_files, repo_id=config.HF_REPO_ID, repo_type=config.HF_REPO_TYPE)
    except Exception:
        log.warning("Dataset repo %s not found yet (first run) — treating as empty", config.HF_REPO_ID)
        return set()
    done = set()
    for f in files:
        if f.startswith(prefix) and f.endswith(suffix):
            done.add(f[len(prefix):-len(suffix)])
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="YYYY-MM-DD, inclusive")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD, inclusive")
    parser.add_argument("--time-budget-hours", type=float, default=DEFAULT_TIME_BUDGET_HOURS)
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    symbol = config.SYMBOL_UM

    api = get_api()
    _with_hf_retry(api.create_repo, repo_id=config.HF_REPO_ID, repo_type=config.HF_REPO_TYPE, exist_ok=True)
    done = completed_dates(api, symbol)
    log.info("%d day(s) already present in %s — will be skipped", len(done), config.HF_REPO_ID)

    remaining = [d for d in _daterange(start, end) if d.isoformat() not in done]
    log.info("%d day(s) remaining out of %d requested", len(remaining), len(list(_daterange(start, end))))

    run_start = time.monotonic()
    budget_seconds = args.time_budget_hours * 3600
    processed = skipped_empty = failed = 0

    for day in remaining:
        elapsed = time.monotonic() - run_start
        if elapsed > budget_seconds:
            log.info("Time budget (%.1fh) reached after %d day(s) this run — stopping cleanly. "
                     "Re-run this script with the same --start/--end to resume from here.",
                     args.time_budget_hours, processed)
            break

        log.info("Processing %s (%.1fh elapsed this run)", day, elapsed / 3600)
        try:
            kaggle_downloader.ensure_day_files(day)
            df = build_day(day)
            if df is None:
                log.warning("No primary data for %s — nothing to upload, moving on", day)
                skipped_empty += 1
                continue

            _with_hf_retry(upload_daily_features, df, symbol, day.isoformat(), api=api)
            processed += 1
            time.sleep(INTER_DAY_PACING_SECONDS)
        except Exception:
            log.exception("Failed to process %s — leaving it unmarked so a future run retries it", day)
            failed += 1
        finally:
            kaggle_downloader.cleanup_day_files(day)

    total_elapsed = time.monotonic() - run_start
    log.info("Run finished: %d uploaded, %d empty/no-data, %d failed, %.1fh elapsed",
              processed, skipped_empty, failed, total_elapsed / 3600)
    if processed + skipped_empty < len(remaining):
        log.info("%d day(s) still remain — schedule another run (e.g. via Kaggle's notebook "
                  "Scheduler) with the same --start/--end to continue.",
                  len(remaining) - processed - skipped_empty)


if __name__ == "__main__":
    main()
