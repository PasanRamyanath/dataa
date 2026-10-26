"""
Live news/on-chain event ingestion per plan §9.1 (free-tier only):

  - RSS headline feeds (CoinDesk, CoinTelegraph, The Block, Decrypt), polled
    every 10s with conditional GET (ETag / If-Modified-Since) so a poll that
    finds nothing new costs almost nothing.
  - GDELT's global news event/tone stream, polled every 15 minutes (GDELT's
    own update cadence) as a slower macro-sentiment drift signal.
  - A free-tier on-chain "whale alert"-style large-transfer feed — public
    blockchain data, not scraped text.

Explicitly excluded (§9.1): CryptoPanic (free tier discontinued early 2026)
and any X/Twitter access. See config.EXCLUDED_NEWS_SOURCES.

Every source writes normalized event dicts as append-only JSONL, one file
per UTC day per source, under config.NEWS_DATA_ROOT — this is the ingestion
layer inference/sentiment_scorer.py tails to do event-triggered scoring.
Conditional-GET ETags and per-source dedup state (seen entry IDs) persist to
config.NEWS_STATE_ROOT so a restarted process doesn't re-fetch/re-append
everything it already saw.

Each poller is independently fault-tolerant: a failing/unreachable source
(network error, missing API key, feed schema change) is logged and skipped
for that cycle rather than crashing the whole ingestion loop — the same
graceful-degradation philosophy models/train_all.py uses for training
members, applied here to live data sources instead.

Usage:
    python -m inference.news_feed                  # run forever
    python -m inference.news_feed --once            # single poll cycle of every source (testing)
    python -m inference.news_feed --skip whale_alert # e.g. no API key available
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("inference.news_feed")

HTTP_TIMEOUT_SECONDS = 10


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _event_id(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()


def _day_path(root: Path, source: str, day: datetime | None = None) -> Path:
    day = day or _now()
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{source}_{day.strftime('%Y-%m-%d')}.jsonl"


def _append_events(source: str, events: list[dict]) -> None:
    if not events:
        return
    path = _day_path(config.NEWS_DATA_ROOT, source)
    with open(path, "a", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, default=str) + "\n")
    log.info("%s: appended %d new event(s) to %s", source, len(events), path)


class _StateStore:
    """Small per-source JSON state file: RSS ETag/Last-Modified cache plus a
    bounded set of recently-seen event IDs (for sources without a native
    conditional-GET mechanism, e.g. GDELT/whale-alert)."""

    def __init__(self, source: str, max_seen: int = 5000):
        config.NEWS_STATE_ROOT.mkdir(parents=True, exist_ok=True)
        self.path = config.NEWS_STATE_ROOT / f"{source}.json"
        self.max_seen = max_seen
        self.data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                log.warning("Corrupt state file %s — starting fresh", self.path)
        return {"etag": None, "last_modified": None, "seen_ids": []}

    def save(self) -> None:
        self.data["seen_ids"] = self.data["seen_ids"][-self.max_seen:]
        self.path.write_text(json.dumps(self.data), encoding="utf-8")

    def is_seen(self, event_id: str) -> bool:
        return event_id in self.data["seen_ids"]

    def mark_seen(self, event_id: str) -> None:
        self.data["seen_ids"].append(event_id)


def _parse_rss_fallback(xml_bytes: bytes) -> list[dict]:
    import xml.etree.ElementTree as ET
    entries = []
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return entries
    for item in root.iter():
        tag = item.tag.split("}")[-1].lower()
        if tag not in ("item", "entry"):
            continue
        rec = {}
        for child in item:
            ctag = child.tag.split("}")[-1].lower()
            text = (child.text or "").strip()
            if ctag in ("guid", "id") and text:
                rec["id"] = text
            elif ctag == "title" and text:
                rec["title"] = text
            elif ctag in ("description", "summary", "content") and text and "summary" not in rec:
                rec["summary"] = text
            elif ctag == "link":
                href = child.attrib.get("href", "").strip()
                rec["link"] = href or text
            elif ctag in ("pubdate", "published", "updated") and text:
                rec["published"] = text
        entries.append(rec)
    return entries


def poll_rss(source: str, url: str) -> list[dict]:
    """One conditional-GET poll of a single RSS feed. Returns only entries
    not already recorded in this source's state (by entry guid/link hash)."""
    try:
        import feedparser
    except ImportError:
        feedparser = None

    state = _StateStore(f"rss_{source}")
    headers = {}
    if state.data.get("etag"):
        headers["If-None-Match"] = state.data["etag"]
    if state.data.get("last_modified"):
        headers["If-Modified-Since"] = state.data["last_modified"]

    try:
        resp = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT_SECONDS)
    except requests.RequestException as e:
        log.warning("RSS poll failed for %s: %s", source, e)
        return []

    if resp.status_code == 304:
        return []  # conditional GET: nothing new, cheapest possible outcome
    if resp.status_code != 200:
        log.warning("RSS poll for %s returned HTTP %d", source, resp.status_code)
        return []

    state.data["etag"] = resp.headers.get("ETag")
    state.data["last_modified"] = resp.headers.get("Last-Modified")

    entries = feedparser.parse(resp.content).entries if feedparser is not None else _parse_rss_fallback(resp.content)
    new_events = []
    for entry in entries:
        guid = entry.get("id") or entry.get("link", "") + entry.get("title", "")
        event_id = _event_id(source, guid)
        if state.is_seen(event_id):
            continue
        state.mark_seen(event_id)
        published = entry.get("published") or entry.get("updated")
        new_events.append({
            "source": source, "kind": "rss_headline", "event_id": event_id,
            "ts": _now().isoformat(), "published": published,
            "title": entry.get("title", ""), "summary": entry.get("summary", ""),
            "url": entry.get("link", ""),
        })

    state.save()
    _append_events(f"rss_{source}", new_events)
    return new_events


def poll_gdelt() -> list[dict]:
    """GDELT DOC 2.0 API article-list query — free, no API key. 15-minute
    cadence macro sentiment/tone signal (plan §9.1)."""
    params = {
        "query": config.GDELT_QUERY, "mode": "artlist", "format": "json",
        "maxrecords": 250, "sort": "datedesc", "timespan": "15min",
    }
    try:
        resp = requests.get(config.GDELT_DOC_API_URL, params=params, timeout=HTTP_TIMEOUT_SECONDS)
        resp.raise_for_status()
        payload = resp.json()
    except (requests.RequestException, ValueError) as e:
        log.debug("GDELT poll skipped/rate-limited: %s", e)
        return []

    state = _StateStore("gdelt")
    new_events = []
    for article in payload.get("articles", []):
        url = article.get("url", "")
        event_id = _event_id("gdelt", url)
        if not url or state.is_seen(event_id):
            continue
        state.mark_seen(event_id)
        new_events.append({
            "source": "gdelt", "kind": "gdelt_article", "event_id": event_id,
            "ts": _now().isoformat(), "seendate": article.get("seendate"),
            "title": article.get("title", ""), "url": url,
            "domain": article.get("domain"), "tone": article.get("tone"),
            "language": article.get("language"),
        })

    state.save()
    _append_events("gdelt", new_events)
    return new_events


def poll_whale_alert() -> list[dict]:
    """Free-tier on-chain large-transfer feed (plan §9.1). Requires
    WHALE_ALERT_API_KEY (config.get_whale_alert_api_key()) — logs and
    returns [] rather than raising when unset, since this is one of three
    §9.1 sources and not required to run the other two."""
    api_key = config.get_whale_alert_api_key()
    if not api_key:
        log.info("WHALE_ALERT_API_KEY not set — skipping whale-alert source (RSS/GDELT still run)")
        return []

    state = _StateStore("whale_alert")
    params = {"api_key": api_key, "min_value": config.WHALE_ALERT_MIN_USD, "start": int(time.time()) - 3600}
    try:
        resp = requests.get(config.WHALE_ALERT_API_URL, params=params, timeout=HTTP_TIMEOUT_SECONDS)
        resp.raise_for_status()
        payload = resp.json()
    except (requests.RequestException, ValueError) as e:
        log.warning("Whale-alert poll failed: %s", e)
        return []

    new_events = []
    for txn in payload.get("transactions", []):
        event_id = _event_id("whale_alert", str(txn.get("hash", "")), str(txn.get("id", "")))
        if state.is_seen(event_id):
            continue
        state.mark_seen(event_id)
        new_events.append({
            "source": "whale_alert", "kind": "large_transfer", "event_id": event_id,
            "ts": _now().isoformat(), "blockchain": txn.get("blockchain"),
            "symbol": txn.get("symbol"), "amount_usd": txn.get("amount_usd"),
            "from_type": (txn.get("from") or {}).get("owner_type"),
            "to_type": (txn.get("to") or {}).get("owner_type"),
        })

    state.save()
    _append_events("whale_alert", new_events)
    return new_events


SOURCES = {
    **{f"rss_{name}": (poll_rss, config.RSS_POLL_INTERVAL_SECONDS, (name, url))
       for name, url in config.RSS_FEEDS.items()},
    "gdelt": (poll_gdelt, config.GDELT_POLL_INTERVAL_SECONDS, ()),
    "whale_alert": (poll_whale_alert, config.WHALE_ALERT_POLL_INTERVAL_SECONDS, ()),
}


def poll_all_once(skip: list[str]) -> dict[str, int]:
    counts = {}
    for name, (fn, _interval, args) in SOURCES.items():
        if name in skip:
            continue
        try:
            events = fn(*args)
            counts[name] = len(events)
        except Exception:
            log.exception("Source %s raised unexpectedly — skipping this cycle", name)
            counts[name] = -1
    return counts


def run_forever(skip: list[str], max_cycles: int | None = None) -> None:
    next_due = {name: 0.0 for name in SOURCES if name not in skip}
    cycles = 0
    while max_cycles is None or cycles < max_cycles:
        now = time.monotonic()
        for name, (fn, interval, args) in SOURCES.items():
            if name in skip or now < next_due.get(name, 0.0):
                continue
            try:
                fn(*args)
            except Exception:
                log.exception("Source %s raised unexpectedly — skipping this cycle", name)
            next_due[name] = now + interval
        cycles += 1
        time.sleep(1.0)  # tick at 1Hz; each source only actually fires at its own interval above


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="run a single poll cycle of every non-skipped source and exit")
    parser.add_argument("--skip", nargs="*", default=[], help=f"source names to skip, e.g. whale_alert. Available: {list(SOURCES)}")
    parser.add_argument("--max-cycles", type=int, default=None, help="stop after N scheduler ticks (testing)")
    args = parser.parse_args()

    if args.once:
        counts = poll_all_once(args.skip)
        log.info("Single poll cycle complete: %s", counts)
    else:
        run_forever(args.skip, args.max_cycles)


if __name__ == "__main__":
    main()
