"""
Targeted (non-crawling) downloader for the exact Binance Vision + Bybit
files that build_dataset.build_day() needs for ONE specific calendar day,
fetched directly into the same on-disk layout paths.py already expects —
so build_day() and every parser in parsers/ run completely unmodified
whether the files came from the local F:\\...\\data sample folders or were
just downloaded fresh into a Kaggle temp directory.

Unlike binance_btc_usd_downloader.py / bybit_btc_usd_downloader.py (which
crawl the whole bucket/site to discover whatever the LATEST sample happens
to be), this constructs each URL directly from the documented Binance
Vision / Bybit naming convention for a SPECIFIC day — no directory listing
needed. That's both far fewer HTTP requests per day and the only way to
backfill an arbitrary historical date range on purpose, rather than
opportunistically grabbing "whatever's newest".

URL construction never hardcodes a second copy of the site's folder
layout: each helper takes the exact local Path that paths.py already
computes, strips the configured root (config.DATA_ROOT / TRADING_ROOT /
etc.), and prepends the matching public base URL — so paths.py remains the
single source of truth for the on-disk/on-site layout convention.
"""
from __future__ import annotations

import logging
import time
from datetime import date
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter

import config
import contracts
import paths
from parsers.binance_parser import MARKET_SPOT, MARKET_UM

log = logging.getLogger("kaggle_downloader")

BINANCE_BASE_URL = "https://data.binance.vision"
BYBIT_BASE_URL = "https://public.bybit.com"

MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 2.0
REQUEST_TIMEOUT = 30
FUNDING_RATE_LOOKBACK_MONTHS = 6  # matches read_funding_rate_for_day's own default

# Months confirmed to have no published fundingRate file yet, cached for
# the lifetime of the process so a multi-hundred-day backfill run doesn't
# re-request the same known-missing month once per day it walks past.
_CONFIRMED_MISSING_FUNDING_MONTHS: set[tuple[str, str, int, int]] = set()


def _session() -> requests.Session:
    s = requests.Session()
    # max_retries=0: retry/backoff is handled explicitly in _get so a 404
    # (file genuinely not published — routine, not an error) is never
    # retried, while 429/5xx are retried with real backoff.
    adapter = HTTPAdapter(max_retries=0, pool_connections=8, pool_maxsize=8)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({"User-Agent": "Mozilla/5.0 (compatible; kaggle-feature-pipeline/1.0)"})
    return s


_SESSION = _session()


def _get(url: str) -> requests.Response | None:
    """GETs a URL with exponential backoff on transient errors. Returns
    None (never raises) for a definitive 404 — Binance/Bybit routinely
    have no file for a given day (before a symbol launched, an unpublished
    current month, a stale/discontinued source like bookTicker), which the
    rest of the pipeline already treats as a normal "source unavailable for
    this day" case rather than an error."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = _SESSION.get(url, timeout=REQUEST_TIMEOUT, stream=True)
        except requests.RequestException as e:
            if attempt == MAX_RETRIES - 1:
                log.warning("Giving up on %s after %d attempts: %s", url, MAX_RETRIES, e)
                return None
            time.sleep(RETRY_BACKOFF_SECONDS * (2 ** attempt))
            continue

        if resp.status_code == 404:
            resp.close()
            return None
        if resp.status_code == 200:
            return resp
        if resp.status_code == 429 or resp.status_code >= 500:
            resp.close()
            if attempt == MAX_RETRIES - 1:
                log.warning("Giving up on %s after %d attempts (HTTP %d)", url, MAX_RETRIES, resp.status_code)
                return None
            time.sleep(RETRY_BACKOFF_SECONDS * (2 ** attempt))
            continue
        resp.close()
        log.warning("Unexpected HTTP %d for %s", resp.status_code, url)
        return None
    return None


def _download_to(url: str, local_path: Path, fetch_checksum: bool = False) -> bool:
    """Skips the request entirely if the file already exists locally with
    non-zero size — the common case for every retried/resumed day, since a
    day's raw files are only deleted AFTER that day is fully processed and
    uploaded (see cleanup_day_files).

    When `fetch_checksum` is set, also best-effort downloads the sibling
    `.CHECKSUM` file: binance_parser.py's readers call utils.verify_checksum()
    and raise on a mismatch, but only actually verify anything if that
    sidecar is present on disk — since a network download (unlike the local
    sample files this pipeline was first built against) can genuinely land
    corrupted, fetching it here is what makes that check load-bearing on
    Kaggle instead of silently skipped."""
    if local_path.exists() and local_path.stat().st_size > 0:
        return True
    resp = _get(url)
    if resp is None:
        return False
    local_path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a .part sidecar and rename only on success: a hard kill
    # mid-download (Kaggle's 12h cutoff, a lost connection) then leaves
    # only an orphaned .part file, never a truncated file at the real path
    # that a parser would try to unzip and fail on.
    tmp_path = local_path.with_name(local_path.name + ".part")
    try:
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=256 * 1024):
                if chunk:
                    f.write(chunk)
    except BaseException:
        # A dropped stream must not strand a partial .part on temp disk —
        # cleanup_day_files only knows the final paths.
        tmp_path.unlink(missing_ok=True)
        raise
    finally:
        resp.close()
    if not tmp_path.exists() or tmp_path.stat().st_size == 0:
        tmp_path.unlink(missing_ok=True)
        return False
    tmp_path.replace(local_path)

    if fetch_checksum:
        checksum_path = local_path.with_suffix(local_path.suffix + ".CHECKSUM")
        if not checksum_path.exists():
            checksum_resp = _get(url + ".CHECKSUM")
            if checksum_resp is not None:
                checksum_path.write_bytes(checksum_resp.content)
                checksum_resp.close()
    return True


def _url_for_local_path(local_path: Path, root: Path, base_url: str, url_prefix: str) -> str:
    rel = local_path.relative_to(root)
    return f"{base_url}/{url_prefix}/{rel.as_posix()}"


def _binance_url(local_path: Path) -> str:
    return _url_for_local_path(local_path, config.DATA_ROOT, BINANCE_BASE_URL, "data")


def ensure_funding_rate_months(market: str, symbol: str, day: date, lookback: int = FUNDING_RATE_LOOKBACK_MONTHS) -> None:
    """Ensures read_funding_rate_for_day()'s backward month-walk has
    something to find: attempts (once per process, per month — see the
    module-level cache) to download each month from `day`'s own month back
    `lookback` months, stopping as soon as one is found locally (freshly
    downloaded or already present from an earlier day in this same run) —
    the reader's own walk will stop at that exact same point."""
    year, month = day.year, day.month
    for _ in range(lookback + 1):
        local_path = paths.funding_rate_path(market, symbol, year, month)
        if local_path.exists() and local_path.stat().st_size > 0:
            return
        key = (market, symbol, year, month)
        if key not in _CONFIRMED_MISSING_FUNDING_MONTHS:
            if _download_to(_binance_url(local_path), local_path, fetch_checksum=True):
                return
            _CONFIRMED_MISSING_FUNDING_MONTHS.add(key)
        month -= 1
        if month == 0:
            month, year = 12, year - 1


def _daily_binance_targets(day: date) -> list[Path]:
    front_contract, back_contract = contracts.front_and_back_quarterly(config.QUARTERLY_BASE_UM, day)
    return [
        paths.agg_trades_path(MARKET_UM, config.SYMBOL_UM, day),
        paths.agg_trades_path(MARKET_SPOT, config.SYMBOL_SPOT, day),
        paths.book_depth_path(MARKET_UM, config.SYMBOL_UM, day),
        paths.book_ticker_path(MARKET_UM, config.SYMBOL_UM, day),
        paths.metrics_path(MARKET_UM, config.SYMBOL_UM, day),
        paths.index_price_klines_path(MARKET_UM, config.SYMBOL_UM, "1m", day),
        paths.mark_price_klines_path(MARKET_UM, config.SYMBOL_UM, "1m", day),
        paths.premium_index_klines_path(MARKET_UM, config.SYMBOL_UM, "1m", day),
        paths.liquidation_snapshot_path(config.SYMBOL_CM, day),
        paths.bvol_index_path(config.BVOL_SYMBOL, day),
        # Term-structure: front + back quarterly delivery-futures contracts,
        # computed deterministically from the day's date (contracts.py) —
        # no directory listing needed. Genuinely absent before quarterly
        # futures existed for this base; build_day/feature_engine.
        # term_structure already fall back to a null schema for those days.
        paths.mark_price_klines_path(MARKET_UM, front_contract.symbol, "1m", day),
        paths.mark_price_klines_path(MARKET_UM, back_contract.symbol, "1m", day),
    ]


def _daily_bybit_index_targets(day: date) -> list[Path]:
    # Live-site-verified (2026-09-12): public.bybit.com/premium_index/BTCUSD/
    # and /spot_index/BTCUSD/ each contain exactly 162 files spanning
    # 2019-10-01..2020-03-10 and nothing since — the tree has been dead site-
    # wide for 6+ years, not merely missing for a few unlucky days. Any day
    # this pipeline actually targets will 404 unconditionally, so these are
    # intentionally excluded from the download targets (see ensure_day_files/
    # cleanup_day_files below) to avoid burning 2 guaranteed-404 requests per
    # day forever. read_bybit_index() already returns None for a missing
    # file, so add_bybit_index_features' null-fallback branch (5 always-null
    # columns: bybit_premium_index, bybit_basis_pct,
    # cross_exch_basis_divergence[_ma_180/_ma_2880]) is exercised exactly the
    # same way with or without ever attempting these downloads.
    return [
        paths.bybit_premium_index_path(config.BYBIT_INDEX_SYMBOL, day),
        paths.bybit_spot_index_path(config.BYBIT_INDEX_SYMBOL, day),
    ]


def ensure_day_files(day: date) -> None:
    """Downloads every raw file build_dataset.build_day(day) will try to
    read — mirrors that function's own read_* calls exactly, so nothing
    extra is fetched (e.g. no cm bookDepth/bookTicker/metrics, since
    build_day never reads those). A no-op network-wise for any file
    already on disk, so re-running this for an already-downloaded day
    costs only stat() calls."""
    for local_path in _daily_binance_targets(day):
        _download_to(_binance_url(local_path), local_path, fetch_checksum=True)

    bybit_path = paths.bybit_trading_path(config.BYBIT_SYMBOL, day)
    _download_to(_url_for_local_path(bybit_path, config.TRADING_ROOT, BYBIT_BASE_URL, "trading"), bybit_path)

    # premium_index/ and spot_index/ are deliberately NOT downloaded here —
    # see _daily_bybit_index_targets' docstring: the live site has had no
    # files under either tree since 2020-03-10, so every attempt for any day
    # this pipeline targets would be a guaranteed 404. build_dataset.py's
    # read_bybit_index() calls already handle the resulting missing files by
    # returning None, which add_bybit_index_features treats as "source
    # unavailable" exactly as if the download had been attempted and 404'd.

    ensure_funding_rate_months(MARKET_UM, config.SYMBOL_UM, day)


def cleanup_day_files(day: date) -> None:
    """Deletes this day's raw downloaded files after it has been fully
    processed (successfully uploaded, or confirmed to have no primary
    data), to keep Kaggle's ~57GB temp disk bounded regardless of how many
    days a single run processes. Monthly fundingRate files are
    deliberately NOT deleted — they're a few KB each and reused by every
    other day in the same month."""
    for local_path in _daily_binance_targets(day):
        local_path.unlink(missing_ok=True)
        Path(str(local_path) + ".CHECKSUM").unlink(missing_ok=True)
    paths.bybit_trading_path(config.BYBIT_SYMBOL, day).unlink(missing_ok=True)
    for local_path in _daily_bybit_index_targets(day):
        local_path.unlink(missing_ok=True)
