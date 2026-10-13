"""
Polite full-text crypto-news scraper: historical backfill + live mode.

Why a scraper instead of a news API: the free news APIs left after
CryptoPanic's free tier died (plan §9.1) give headlines only and little or
no history. The accuracy of a news signal is dominated by two things this
module is built around:

1. TRAINING HISTORY. A model can only learn how much weight to give news if
   the training window (~24 months of bars) has news features too. Live-only
   collection would leave the models with no labeled examples for months.
   `backfill` crawls each outlet's XML sitemaps (discovered from robots.txt,
   pruned by <lastmod> so old sub-sitemaps are skipped) over any date range;
   `gdelt-backfill` adds GDELT DOC API history, whose `seendate` is a
   genuinely point-in-time "first seen" timestamp.
2. POINT-IN-TIME TIMESTAMPS. Each article records published_ts (from the
   news-sitemap <news:publication_date>, article:published_time meta tag or
   JSON-LD datePublished — never the modified date), first_seen_ts (when we
   fetched it) and available_ts, the moment the pipeline may use it: the
   real first_seen_ts for live articles, published_ts +
   config.NEWS_BACKFILL_AVAILABILITY_LAG_SECONDS for backfilled ones.
   news_pipeline.features only ever uses available_ts.

Full text is extracted because the LLM scorer reads the body, not only the
headline — headline-only sentiment misses the substance of most stories
("X announces..." headlines are neutral; the body says whether it is a $5M
or a $5B event). Extraction is a consensus ensemble (extract_article):
up to four candidates — the publisher's own JSON-LD articleBody,
trafilatura in precision mode, trafilatura in recall mode, and a stdlib
paragraph pass — and the one most closely confirmed by another extractor
(best pairwise bag-of-words F1) wins. Trafilatura is the best single extractor in the independent
benchmarks (SIGIR 2023 comparison of 8 datasets / 3,985 pages; ~0.93-0.95
F1 on news), and the same study found ensembles beat any single tool;
agreement voting also rejects the classic single-extractor failures (a
teaser-only JSON-LD body, a related-articles rail picked up in recall
mode). Fundus (F1 ~0.98 on its news benchmark) was evaluated but only
ships parsers for mainstream publishers — none of config.SCRAPER_SITES.

Fetching is concurrent ACROSS domains (config.SCRAPER_MAX_WORKERS threads,
jobs sharded by domain) and strictly sequential WITHIN a domain, so a
GDELT backfill spanning hundreds of outlets finishes ~workers x faster
while no single site ever sees more than one in-flight request or a rate
above its limit.

Politeness is not optional: robots.txt is honored (Disallow and
Crawl-delay), each domain gets at most one request per
max(Crawl-delay, config.SCRAPER_MIN_DOMAIN_INTERVAL_SECONDS), the
User-Agent identifies the bot, 429/5xx back off, and paywalled or blocked
pages are recorded headline-only rather than worked around. There is no
anti-bot evasion here, and none should be added. Check each outlet's terms
before running a large backfill.

Storage: append-only JSONL under config.NEWS_ARTICLES_ROOT/raw/, one file
per available_ts day, deduplicated by URL hash. Cross-outlet near-duplicate
story clustering happens in news_pipeline.features (it needs the whole
time-ordered set, which backfill does not produce in order).

Usage:
    python -m news_pipeline.scraper backfill --start 2024-09-01 --end 2026-09-01 [--sites coindesk,decrypt] [--max-articles N]
    python -m news_pipeline.scraper gdelt-backfill --start 2026-07-01 --end 2026-09-01
    python -m news_pipeline.scraper live [--max-cycles N]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import re
import threading
import time
import urllib.robotparser
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("news_pipeline.scraper")

MIN_BODY_CHARS = 200  # shorter than this = paywall/teaser/blocked page -> headline-only record


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------
def parse_dt(value) -> datetime | None:
    """Parses ISO-8601 (with Z/offset/naive-as-UTC), RFC-2822 (RSS pubDate),
    GDELT's 20240101T120000Z and bare YYYY-MM-DD into an aware UTC datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    s = str(value).strip()
    if not s:
        return None
    m = re.fullmatch(r"(\d{4})(\d{2})(\d{2})T?(\d{2})(\d{2})(\d{2})Z?", s)
    if m:
        return datetime(*map(int, m.groups()), tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError, IndexError):
        return None


def _iso(dt: datetime | None) -> str | None:
    return dt.astimezone(timezone.utc).isoformat() if dt else None


def canonical_url(url: str) -> str:
    p = urlparse(url.strip())
    return urlunparse((p.scheme.lower() or "https", p.netloc.lower(), p.path.rstrip("/") or "/", "", "", ""))


def url_id(url: str) -> str:
    return hashlib.sha1(canonical_url(url).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Polite fetcher
# ---------------------------------------------------------------------------
class PoliteFetcher:
    """requests wrapper enforcing robots.txt, per-domain rate limits,
    retries with backoff and a response-size cap. Thread-safe: any number of
    threads may share one fetcher — robots.txt is loaded once per origin,
    each origin has at most one request in flight, and each thread gets its
    own requests.Session (Session isn't documented thread-safe) unless a
    session is injected."""

    def __init__(self, user_agent: str = config.SCRAPER_USER_AGENT,
                 min_interval: float = config.SCRAPER_MIN_DOMAIN_INTERVAL_SECONDS,
                 timeout: float = config.SCRAPER_TIMEOUT_SECONDS,
                 max_retries: int = config.SCRAPER_MAX_RETRIES,
                 session=None, sleep=time.sleep):
        self._headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
        self._shared_session = session
        if session is not None and hasattr(session, "headers"):
            session.headers.update(self._headers)
        self._tls = threading.local()
        self.user_agent = user_agent
        self.min_interval = min_interval
        self.timeout = timeout
        self.max_retries = max_retries
        self._sleep = sleep
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self._robots_sitemaps: dict[str, list[str]] = {}
        self._last_request: dict[str, float] = {}
        self._lock = threading.Lock()
        self._robots_locks: dict[str, threading.Lock] = {}
        self._inflight_locks: dict[str, threading.Lock] = {}

    @property
    def session(self):
        if self._shared_session is not None:
            return self._shared_session
        s = getattr(self._tls, "session", None)
        if s is None:
            import requests

            s = requests.Session()
            s.headers.update(self._headers)
            self._tls.session = s
        return s

    def _origin_lock(self, table: dict[str, threading.Lock], origin: str) -> threading.Lock:
        with self._lock:
            return table.setdefault(origin, threading.Lock())

    @staticmethod
    def _origin(url: str) -> str:
        p = urlparse(url)
        return f"{p.scheme}://{p.netloc}"

    def _load_robots(self, origin: str) -> urllib.robotparser.RobotFileParser | None:
        if origin in self._robots:
            return self._robots[origin]
        with self._origin_lock(self._robots_locks, origin):
            if origin in self._robots:  # another thread loaded it while we waited
                return self._robots[origin]
            return self._load_robots_locked(origin)

    def _load_robots_locked(self, origin: str) -> urllib.robotparser.RobotFileParser | None:
        rp = urllib.robotparser.RobotFileParser()
        sitemaps: list[str] = []
        try:
            resp = self._raw_get(origin + "/robots.txt", origin)
            if resp is not None and resp.status_code == 200:
                lines = resp.text.splitlines()
                rp.parse(lines)
                sitemaps = [ln.split(":", 1)[1].strip() for ln in lines if ln.lower().startswith("sitemap:")]
            elif resp is None or resp.status_code in (401, 403):
                rp.disallow_all = True  # unreachable/forbidden = assume full disallow (RFC 9309)
            else:
                rp.allow_all = True  # 404 etc. = no restrictions, per the robots spec
        except Exception:
            log.warning("robots.txt unreachable for %s — treating as disallow-all this run", origin)
            rp.disallow_all = True
        self._robots_sitemaps[origin] = sitemaps
        self._robots[origin] = rp  # published last: its presence means the entry is complete
        return rp

    def allowed(self, url: str) -> bool:
        rp = self._load_robots(self._origin(url))
        return rp.can_fetch(self.user_agent, url)

    def sitemaps_for(self, base_url: str) -> list[str]:
        origin = self._origin(base_url)
        self._load_robots(origin)
        return self._robots_sitemaps.get(origin) or [origin + "/sitemap.xml"]

    def _wait_turn(self, origin: str) -> None:
        interval = self.min_interval
        rp = self._robots.get(origin)
        if rp is not None:
            try:
                delay = rp.crawl_delay(self.user_agent)
            except Exception:
                delay = None
            if delay:
                interval = max(interval, float(delay))
        with self._lock:
            now = time.monotonic()
            wait = self._last_request.get(origin, -1e18) + interval - now
            self._last_request[origin] = now + max(0.0, wait)
        if wait > 0:
            self._sleep(wait)

    def _raw_get(self, url: str, origin: str):
        # one request (including its retry backoff) in flight per origin
        with self._origin_lock(self._inflight_locks, origin):
            return self._raw_get_locked(url, origin)

    def _raw_get_locked(self, url: str, origin: str):
        for attempt in range(self.max_retries):
            self._wait_turn(origin)
            try:
                resp = self.session.get(url, timeout=self.timeout, stream=True)
            except Exception as exc:
                log.debug("GET %s failed (%s), attempt %d", url, exc, attempt + 1)
                self._sleep(2 ** attempt)
                continue
            if resp.status_code == 429 or resp.status_code >= 500:
                retry_after = resp.headers.get("Retry-After", "")
                resp.close()  # stream=True: hand the unread connection back to the pool
                self._sleep(float(retry_after) if retry_after.isdigit() else 5.0 * (2 ** attempt))
                continue
            content = resp.raw.read(config.SCRAPER_MAX_BYTES + 1, decode_content=True) if getattr(resp, "raw", None) is not None and hasattr(resp.raw, "read") else resp.content
            if content and len(content) > config.SCRAPER_MAX_BYTES:
                log.warning("GET %s exceeded %d bytes — skipped", url, config.SCRAPER_MAX_BYTES)
                resp.close()
                return None
            resp._content = content
            return resp
        return None

    def get(self, url: str):
        """Returns a response with status 200, or None (robots-disallowed,
        error, non-200, too large)."""
        origin = self._origin(url)
        if not self.allowed(url):
            log.debug("robots.txt disallows %s", url)
            return None
        resp = self._raw_get(url, origin)
        if resp is None or resp.status_code != 200:
            return None
        return resp


# ---------------------------------------------------------------------------
# Sitemaps / RSS
# ---------------------------------------------------------------------------
def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_sitemap(data: bytes) -> tuple[str, list[dict]]:
    """Returns ("index", [{loc, lastmod}]) for a sitemap index or
    ("urlset", [{loc, lastmod, news_published, news_title}]) for a urlset.
    Handles gzip and the Google News sitemap extension."""
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    root = ET.fromstring(data)
    kind = "index" if _local(root.tag) == "sitemapindex" else "urlset"
    entries = []
    for node in root:
        if _local(node.tag) not in ("sitemap", "url"):
            continue
        entry: dict = {"loc": None, "lastmod": None, "news_published": None, "news_title": None}
        for child in node.iter():
            name = _local(child.tag)
            text = (child.text or "").strip()
            if name == "loc" and entry["loc"] is None:
                entry["loc"] = text
            elif name == "lastmod":
                entry["lastmod"] = text
            elif name == "publication_date":
                entry["news_published"] = text
            elif name == "title" and child is not node:
                entry["news_title"] = text
        if entry["loc"]:
            entries.append(entry)
    return kind, entries


def iter_sitemap_articles(fetcher: PoliteFetcher, sitemap_url: str, start: datetime, end: datetime,
                          article_pattern: str | None = None, _depth: int = 0):
    """Walks a sitemap (index) tree yielding article entries whose
    publication date (news:publication_date, else lastmod) lies in
    [start, end). Child sitemaps whose lastmod is before `start` are pruned —
    a sub-sitemap can't contain anything newer than its own lastmod."""
    if _depth > 4:
        return
    resp = fetcher.get(sitemap_url)
    if resp is None:
        return
    try:
        kind, entries = parse_sitemap(resp.content)
    except ET.ParseError:
        log.warning("unparseable sitemap %s", sitemap_url)
        return
    pattern = re.compile(article_pattern) if article_pattern else None
    if kind == "index":
        # newest first so a --max-articles cap keeps the most recent history
        entries.sort(key=lambda e: parse_dt(e["lastmod"]) or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        for e in entries:
            lastmod = parse_dt(e["lastmod"])
            if lastmod is not None and lastmod < start:
                continue
            yield from iter_sitemap_articles(fetcher, e["loc"], start, end, article_pattern, _depth + 1)
        return
    for e in entries:
        if pattern and not pattern.search(urlparse(e["loc"]).path):
            continue
        published = parse_dt(e["news_published"]) or parse_dt(e["lastmod"])
        if published is None or not (start <= published < end):
            continue
        e["sitemap_published_ts"] = published
        e["published_from_news_ext"] = e["news_published"] is not None
        yield e


def parse_rss(data: bytes) -> list[dict]:
    """[{loc, published, title}] from an RSS/Atom feed."""
    try:
        import feedparser

        feed = feedparser.parse(data)
        return [{"loc": e.get("link"), "published": e.get("published") or e.get("updated"), "title": e.get("title")}
                for e in feed.entries if e.get("link")]
    except ImportError:
        pass
    root = ET.fromstring(data)
    out = []
    for item in root.iter():
        if _local(item.tag) not in ("item", "entry"):
            continue
        rec = {"loc": None, "published": None, "title": None}
        for child in item:
            name = _local(child.tag)
            if name == "link":
                rec["loc"] = (child.text or "").strip() or child.attrib.get("href")
            elif name in ("pubDate", "published", "updated") and rec["published"] is None:
                rec["published"] = (child.text or "").strip()
            elif name == "title":
                rec["title"] = (child.text or "").strip()
        if rec["loc"]:
            out.append(rec)
    return out


# ---------------------------------------------------------------------------
# Article extraction
# ---------------------------------------------------------------------------
class _MetaParser(HTMLParser):
    """Stdlib-only metadata + paragraph extractor (fallback when trafilatura
    isn't installed, and the source of precise timestamps either way —
    trafilatura's own date field is day-precision)."""

    _SKIP = {"script", "style", "nav", "footer", "aside", "header", "form", "noscript"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, str] = {}
        self.title = ""
        self.jsonld: list[str] = []
        self.time_datetimes: list[str] = []
        self.paragraphs: list[str] = []
        self._stack: list[str] = []
        self._buf: list[str] = []
        self._in_p = False
        self._in_title = False
        self._in_jsonld = False
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "meta":
            key = (a.get("property") or a.get("name") or a.get("itemprop") or "").lower()
            if key and a.get("content"):
                self.meta.setdefault(key, a["content"])
            return
        if tag == "time" and a.get("datetime"):
            self.time_datetimes.append(a["datetime"])
        if tag == "script" and (a.get("type") or "").lower() == "application/ld+json":
            self._in_jsonld = True
            self._buf = []
            return
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag == "p" and not self._skip_depth:
            self._in_p = True
            self._buf = []

    def handle_endtag(self, tag):
        if tag == "script" and self._in_jsonld:
            self.jsonld.append("".join(self._buf))
            self._in_jsonld = False
            return
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False
        elif tag == "p" and self._in_p:
            text = " ".join("".join(self._buf).split())
            if len(text) > 30:
                self.paragraphs.append(text)
            self._in_p = False

    def handle_data(self, data):
        if self._in_jsonld or self._in_p:
            self._buf.append(data)
        elif self._in_title:
            self.title += data


def _jsonld_published(blobs: list[str]) -> str | None:
    def walk(obj):
        if isinstance(obj, dict):
            if obj.get("datePublished"):
                return obj["datePublished"]
            for v in obj.values():
                r = walk(v)
                if r:
                    return r
        elif isinstance(obj, list):
            for v in obj:
                r = walk(v)
                if r:
                    return r
        return None

    for blob in blobs:
        try:
            r = walk(json.loads(blob))
        except (json.JSONDecodeError, TypeError):
            continue
        if r:
            return r
    return None


_ARTICLE_TYPES = {"newsarticle", "article", "reportagenewsarticle", "analysisnewsarticle", "blogposting"}


def _jsonld_article_body(blobs: list[str]) -> str | None:
    """articleBody of the page's (News)Article JSON-LD node — the
    publisher's own clean body text when present (HTML tags stripped)."""
    def walk(obj):
        if isinstance(obj, dict):
            types = obj.get("@type")
            types = {str(t).lower() for t in (types if isinstance(types, list) else [types])}
            body = obj.get("articleBody")
            if isinstance(body, str) and (types & _ARTICLE_TYPES or not types - {"none"}):
                return body
            for v in obj.values():
                r = walk(v)
                if r:
                    return r
        elif isinstance(obj, list):
            for v in obj:
                r = walk(v)
                if r:
                    return r
        return None

    for blob in blobs:
        try:
            body = walk(json.loads(blob))
        except (json.JSONDecodeError, TypeError):
            continue
        if body:
            body = unescape(re.sub(r"<[^>]+>", " ", body))
            return "\n".join(" ".join(ln.split()) for ln in body.splitlines() if ln.strip())
    return None


def extract_article(html: str, url: str) -> dict:
    """{title, text, published_ts (datetime|None), published_source, extractor}."""
    mp = _MetaParser()
    try:
        mp.feed(html)
    except Exception:
        pass
    published, published_source = None, None
    for source, raw in (("meta:article:published_time", mp.meta.get("article:published_time")),
                        ("jsonld:datePublished", _jsonld_published(mp.jsonld)),
                        ("meta:datepublished", mp.meta.get("datepublished") or mp.meta.get("date")),
                        ("time[datetime]", mp.time_datetimes[0] if mp.time_datetimes else None)):
        published = parse_dt(raw)
        if published:
            published_source = source
            break
    title = (mp.meta.get("og:title") or mp.title or "").strip()
    candidates = {"jsonld": _jsonld_article_body(mp.jsonld) or ""}
    try:
        import trafilatura

        for name, mode in (("trafilatura_precision", {"favor_precision": True}),
                           ("trafilatura_recall", {"favor_recall": True})):
            try:
                candidates[name] = trafilatura.extract(html, url=url, include_comments=False,
                                                       include_tables=False, **mode) or ""
            except Exception:
                log.debug("trafilatura (%s) failed on %s", name, url, exc_info=True)
    except ImportError:
        pass
    rs_text = _rs_trafilatura_extract(html, url)
    if rs_text is not None:
        candidates["rs_trafilatura"] = rs_text
    candidates["stdlib"] = "\n".join(mp.paragraphs)
    text, extractor, agreement = select_consensus_text(candidates)
    return {"title": title, "text": text.strip(), "published_ts": published,
            "published_source": published_source, "extractor": extractor,
            "extract_agreement": round(agreement, 4)}


def _rs_trafilatura_extract(html: str, url: str) -> str | None:
    """main_content from rs-trafilatura's extract_stdin CLI (HTML on stdin,
    JSON on stdout), or None when config.RS_TRAFILATURA_BIN isn't set/usable."""
    binary = config.RS_TRAFILATURA_BIN
    if not binary or not Path(binary).is_file():
        return None
    import subprocess

    try:
        proc = subprocess.run([binary, "--url", url], input=html.encode("utf-8", "replace"),
                              capture_output=True, timeout=config.RS_TRAFILATURA_TIMEOUT_S, check=False)
        if proc.returncode != 0:
            return None
        return json.loads(proc.stdout.decode("utf-8", "replace")).get("main_content") or ""
    except Exception:
        log.debug("rs-trafilatura failed on %s", url, exc_info=True)
        return None


# Tie-break order among near-equal agreement: publisher-supplied body first,
# then the WCXB 2026 benchmark winner (rs-trafilatura, when configured), then
# trafilatura's precision and recall modes, then the stdlib pass.
EXTRACTOR_PRIORITY = ("jsonld", "rs_trafilatura", "trafilatura_precision", "trafilatura_recall", "stdlib")
CONSENSUS_TIE_MARGIN = 0.05
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _bag(text: str) -> Counter:
    return Counter(w.lower() for w in _WORD_RE.findall(text or ""))


def bag_f1(a: Counter, b: Counter) -> float:
    """Bag-of-words F1 between two texts — the token-level metric the
    extraction benchmarks (trafilatura's, Bevendorff et al. 2023) use."""
    if not a or not b:
        return 0.0
    overlap = sum((a & b).values())
    if not overlap:
        return 0.0
    p, r = overlap / sum(a.values()), overlap / sum(b.values())
    return 2 * p * r / (p + r)


def select_consensus_text(candidates: dict[str, str]) -> tuple[str, str, float]:
    """(text, extractor name, agreement) — consensus selection: each
    body-length candidate is scored by its BEST bag-of-words F1 with any
    other candidate (independent confirmation), and the highest-priority
    candidate within CONSENSUS_TIE_MARGIN of the top score wins. Best-match
    rather than mean agreement matters: a boilerplate-padded superset (e.g.
    favor_recall output) half-agrees with everything and would win a mean
    vote over two extractors that agree almost exactly. With two candidates agreement is symmetric, so the
    choice falls to priority unless they disagree badly (F1 < 0.6), in which
    case a benchmark winner (rs-trafilatura, else trafilatura precision) is trusted
    over a possibly-truncated or boilerplate-laden alternative."""
    usable = {k: v.strip() for k, v in candidates.items() if v and len(v.strip()) >= MIN_BODY_CHARS}
    if not usable:
        name, text = max(candidates.items(), key=lambda kv: len(kv[1] or ""), default=("none", ""))
        return (text or "").strip(), (name if text else "none"), 0.0
    if len(usable) == 1:
        name, text = next(iter(usable.items()))
        return text, name, 1.0
    bags = {k: _bag(v) for k, v in usable.items()}
    agree = {k: max(bag_f1(bags[k], bags[o]) for o in usable if o != k) for k in usable}
    ordered = [k for k in EXTRACTOR_PRIORITY if k in usable] + [k for k in usable if k not in EXTRACTOR_PRIORITY]
    if len(usable) == 2 and max(agree.values()) < 0.6:
        pick = next((k for k in ("rs_trafilatura", "trafilatura_precision") if k in usable),
                    max(usable, key=lambda k: len(usable[k])))
        return usable[pick], pick, agree[pick]
    top = max(agree.values())
    pick = next(k for k in ordered if agree[k] >= top - CONSENSUS_TIE_MARGIN)
    return usable[pick], pick, agree[pick]


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------
@dataclass
class ArticleStore:
    """Append-only JSONL store, safe to share across fetch threads: add()
    is serialized, and claim()/release() keep two threads from fetching the
    same URL at once (GDELT often lists one URL in overlapping windows)."""
    root: Path = field(default_factory=lambda: config.NEWS_ARTICLES_ROOT / "raw")

    def __post_init__(self):
        self.root = Path(self.root)
        self._seen: set[str] | None = None
        self._inflight: set[str] = set()
        self._lock = threading.RLock()

    def seen_ids(self) -> set[str]:
        with self._lock:
            if self._seen is None:
                seen = set()
                for path in sorted(self.root.glob("articles_*.jsonl")):
                    with path.open(encoding="utf-8") as fh:
                        for line in fh:
                            try:
                                seen.add(json.loads(line)["article_id"])
                            except (json.JSONDecodeError, KeyError):
                                continue
                self._seen = seen
            return self._seen

    def has(self, url: str) -> bool:
        return url_id(url) in self.seen_ids()

    def claim(self, url: str) -> bool:
        """True if this caller may fetch `url` (not stored, not in flight)."""
        aid = url_id(url)
        with self._lock:
            if aid in self.seen_ids() or aid in self._inflight:
                return False
            self._inflight.add(aid)
            return True

    def release(self, url: str) -> None:
        with self._lock:
            self._inflight.discard(url_id(url))

    def add(self, record: dict) -> bool:
        with self._lock:
            if record["article_id"] in self.seen_ids():
                return False
            self.root.mkdir(parents=True, exist_ok=True)
            day = record["available_ts"][:10]
            with (self.root / f"articles_{day}.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._seen.add(record["article_id"])
            return True


def build_record(url: str, site: str, extracted: dict, *, mode: str, first_seen: datetime,
                 hint_published: datetime | None = None, hint_title: str | None = None,
                 fallback_published: datetime | None = None, fetch_ok: bool = True) -> dict | None:
    """Assembles the stored record and its point-in-time available_ts.
    Precedence for published_ts: news-sitemap/RSS/GDELT hint (dated by the
    publisher's own feed or by GDELT's crawler) > page metadata > fallback
    (a plain sitemap <lastmod>, the least reliable)."""
    if hint_published:
        published, published_source = hint_published, "feed"
    elif extracted.get("published_ts"):
        published, published_source = extracted["published_ts"], extracted.get("published_source")
    else:
        published, published_source = fallback_published, ("sitemap:lastmod" if fallback_published else None)
    if mode == "live":
        # we genuinely saw it at first_seen; a publisher clock claiming a
        # later time must not move availability into the future
        available = first_seen
        if published is None or published > first_seen:
            published = first_seen
    else:
        if published is None:
            return None  # undated backfill article can't be placed point-in-time — drop it
        available = published + timedelta(seconds=config.NEWS_BACKFILL_AVAILABILITY_LAG_SECONDS)
    text = extracted.get("text", "") if fetch_ok else ""
    title = (extracted.get("title") or hint_title or "").strip()
    if not title and not text:
        return None
    return {
        "article_id": url_id(url),
        "url": canonical_url(url),
        "site": site,
        "mode": mode,
        "title": title,
        "text": text if len(text) >= MIN_BODY_CHARS else "",
        "body_available": len(text) >= MIN_BODY_CHARS,
        "published_ts": _iso(published),
        "published_source": published_source,
        "first_seen_ts": _iso(first_seen),
        "available_ts": _iso(available),
        "extractor": extracted.get("extractor"),
        "extract_agreement": extracted.get("extract_agreement"),
    }


def _fetch_and_store(fetcher: PoliteFetcher, store: ArticleStore, url: str, site: str, *, mode: str,
                     hint_published: datetime | None, hint_title: str | None,
                     fallback_published: datetime | None = None,
                     now_fn=lambda: datetime.now(timezone.utc)) -> dict | None:
    if not store.claim(url):
        return None
    try:
        first_seen = now_fn()
        resp = fetcher.get(url)
        if resp is not None:
            extracted = extract_article(resp.text, url)
            fetch_ok = True
        else:
            # blocked/paywalled/robots-disallowed: keep the headline if the feed gave one
            extracted, fetch_ok = {"title": hint_title or "", "text": ""}, False
        record = build_record(url, site, extracted, mode=mode, first_seen=first_seen,
                              hint_published=hint_published, hint_title=hint_title,
                              fallback_published=fallback_published, fetch_ok=fetch_ok)
        if record is not None and store.add(record):
            return record
        return None
    finally:
        store.release(url)


# ---------------------------------------------------------------------------
# Concurrency: parallel across domains, sequential within one
# ---------------------------------------------------------------------------
class ArticleBudget:
    """Thread-safe --max-articles counter. reserve() before a fetch,
    commit(stored) after it, so N concurrent workers can't overshoot."""

    def __init__(self, limit: int | None):
        self.limit = limit
        self.stored = 0
        self._reserved = 0
        self._lock = threading.Lock()

    def reserve(self) -> bool:
        with self._lock:
            if self.limit is not None and self.stored + self._reserved >= self.limit:
                return False
            self._reserved += 1
            return True

    def commit(self, stored: bool) -> None:
        with self._lock:
            self._reserved -= 1
            self.stored += int(stored)

    @property
    def exhausted(self) -> bool:
        with self._lock:
            return self.limit is not None and self.stored >= self.limit


def domain_of(url: str) -> str:
    return urlparse(url).netloc.lower()


def run_domain_sharded(items: list, fn, key=lambda it: domain_of(it["url"]),
                       max_workers: int = config.SCRAPER_MAX_WORKERS) -> list:
    """Applies fn to every item, grouping items by `key` (their domain):
    groups run concurrently on up to max_workers threads, items within a
    group run sequentially in their original order — the per-domain rate
    limit then never makes a worker idle while another domain has work.
    Results come back in group order; an exception in fn is logged and
    yields None for that item only."""
    groups: dict[str, list] = {}
    for it in items:
        groups.setdefault(key(it), []).append(it)
    if not groups:
        return []

    def run_group(group: list) -> list:
        out = []
        for it in group:
            try:
                out.append(fn(it))
            except Exception:
                log.exception("fetch job failed: %s", it.get("url") if isinstance(it, dict) else it)
                out.append(None)
        return out

    workers = max(1, min(max_workers, len(groups)))
    if workers == 1:
        return [r for g in groups.values() for r in run_group(g)]
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="scrape") as ex:
        return [r for res in ex.map(run_group, list(groups.values())) for r in res]


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def backfill(start: datetime, end: datetime, sites: list[str] | None = None, max_articles: int | None = None,
             fetcher: PoliteFetcher | None = None, store: ArticleStore | None = None,
             max_workers: int = config.SCRAPER_MAX_WORKERS) -> int:
    """Crawls every selected site's sitemaps concurrently (one worker per
    site — each site is its own domain, so politeness is unchanged)."""
    fetcher = fetcher or PoliteFetcher()
    store = store or ArticleStore()
    budget = ArticleBudget(max_articles)

    def crawl_site(job: dict) -> int:
        site, spec = job["site"], job["spec"]
        site_n = 0
        for sitemap in fetcher.sitemaps_for(spec["base_url"]):
            for entry in iter_sitemap_articles(fetcher, sitemap, start, end, spec.get("article_pattern")):
                if not budget.reserve():
                    log.info("backfill %s: reached --max-articles=%d", site, max_articles)
                    return site_n
                news_ext = entry["published_from_news_ext"]
                rec = None
                try:
                    rec = _fetch_and_store(fetcher, store, entry["loc"], site, mode="backfill",
                                           hint_published=entry["sitemap_published_ts"] if news_ext else None,
                                           fallback_published=None if news_ext else entry["sitemap_published_ts"],
                                           hint_title=entry.get("news_title"))
                finally:
                    budget.commit(rec is not None)
                site_n += rec is not None
        log.info("backfill %s: %d new articles", site, site_n)
        return site_n

    jobs = [{"site": site, "spec": spec, "url": spec["base_url"]} for site, spec in config.SCRAPER_SITES.items()
            if not sites or site in sites]
    run_domain_sharded(jobs, crawl_site, max_workers=max_workers)
    return budget.stored


def _gdelt_window(fetcher: PoliteFetcher, start: datetime, end: datetime) -> list[dict]:
    import urllib.parse

    params = {"query": f"({config.GDELT_QUERY}) sourcelang:english", "mode": "artlist", "format": "json",
              "maxrecords": "250", "sort": "datedesc",
              "startdatetime": start.strftime("%Y%m%d%H%M%S"), "enddatetime": end.strftime("%Y%m%d%H%M%S")}
    resp = fetcher.get(config.GDELT_DOC_API_URL + "?" + urllib.parse.urlencode(params))
    if resp is None:
        return []
    try:
        return resp.json().get("articles", []) or []
    except ValueError:
        return []


def gdelt_backfill(start: datetime, end: datetime, fetch_bodies: bool = True, max_articles: int | None = None,
                   fetcher: PoliteFetcher | None = None, store: ArticleStore | None = None,
                   step: timedelta = timedelta(hours=6), max_workers: int = config.SCRAPER_MAX_WORKERS) -> int:
    """GDELT DOC 2.0 artlist over [start, end) in `step` windows (the API
    returns <=250 records per call, so windows stay small). seendate is
    GDELT's own first-crawl time — a clean point-in-time timestamp, used as
    published_ts. The DOC API only reaches back ~3 months; sitemap backfill
    covers the rest of the training window. Each window's article bodies
    are fetched domain-sharded in parallel (a window typically spans
    dozens of outlets)."""
    fetcher = fetcher or PoliteFetcher()
    store = store or ArticleStore()
    budget = ArticleBudget(max_articles)

    def fetch_one(job: dict) -> dict | None:
        if not budget.reserve():
            return None
        rec = None
        try:
            site = f"gdelt:{domain_of(job['url'])}"
            if fetch_bodies:
                rec = _fetch_and_store(fetcher, store, job["url"], site, mode="backfill",
                                       hint_published=job["seen"], hint_title=job["title"])
            else:
                rec = build_record(job["url"], site, {"title": job["title"] or "", "text": ""}, mode="backfill",
                                   first_seen=datetime.now(timezone.utc), hint_published=job["seen"], fetch_ok=False)
                rec = rec if rec is not None and store.add(rec) else None
        finally:
            budget.commit(rec is not None)
        return rec

    cursor = start
    while cursor < end and not budget.exhausted:
        window_end = min(cursor + step, end)
        jobs = []
        for art in _gdelt_window(fetcher, cursor, window_end):
            url, seen = art.get("url"), parse_dt(art.get("seendate"))
            if url and seen is not None:
                jobs.append({"url": url, "seen": seen, "title": art.get("title")})
        run_domain_sharded(jobs, fetch_one, max_workers=max_workers if fetch_bodies else 1)
        cursor = window_end
    log.info("gdelt-backfill: %d new articles", budget.stored)
    return budget.stored


def live_poll_once(fetcher: PoliteFetcher | None = None, store: ArticleStore | None = None,
                   sites: list[str] | None = None, now_fn=lambda: datetime.now(timezone.utc),
                   max_workers: int = config.SCRAPER_MAX_WORKERS) -> list[dict]:
    """One pass over every site's RSS feed, sites in parallel; new URLs are
    fetched, extracted and stored with first_seen_ts = now (event-triggered
    scoring picks them up from the store)."""
    fetcher = fetcher or PoliteFetcher()
    store = store or ArticleStore()

    def poll_site(job: dict) -> list[dict]:
        site, spec = job["site"], job["spec"]
        resp = fetcher.get(spec["rss"])
        if resp is None:
            return []
        try:
            items = parse_rss(resp.content)
        except ET.ParseError:
            log.warning("unparseable RSS for %s", site)
            return []
        new = []
        for item in items:
            rec = _fetch_and_store(fetcher, store, item["loc"], site, mode="live",
                                   hint_published=parse_dt(item.get("published")), hint_title=item.get("title"),
                                   now_fn=now_fn)
            if rec is not None:
                new.append(rec)
        return new

    jobs = [{"site": site, "spec": spec, "url": spec["base_url"]} for site, spec in config.SCRAPER_SITES.items()
            if (not sites or site in sites) and spec.get("rss")]
    return [rec for recs in run_domain_sharded(jobs, poll_site, max_workers=max_workers) if recs for rec in recs]


def run_live(max_cycles: int | None = None, poll_seconds: float = config.SCRAPER_LIVE_POLL_SECONDS) -> None:
    fetcher, store = PoliteFetcher(), ArticleStore()
    cycle = 0
    while max_cycles is None or cycle < max_cycles:
        started = time.monotonic()
        try:
            new = live_poll_once(fetcher, store)
            if new:
                log.info("live: %d new articles", len(new))
        except Exception:
            log.exception("live poll failed; continuing")
        cycle += 1
        time.sleep(max(0.0, poll_seconds - (time.monotonic() - started)))


def _date(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backfill")
    b.add_argument("--start", required=True)
    b.add_argument("--end", required=True)
    b.add_argument("--sites", default=None, help="comma-separated subset of config.SCRAPER_SITES")
    b.add_argument("--max-articles", type=int, default=None)
    g = sub.add_parser("gdelt-backfill")
    g.add_argument("--start", required=True)
    g.add_argument("--end", required=True)
    g.add_argument("--headlines-only", action="store_true", help="don't fetch article bodies")
    g.add_argument("--max-articles", type=int, default=None)
    lv = sub.add_parser("live")
    lv.add_argument("--max-cycles", type=int, default=None)
    args = parser.parse_args()

    if args.cmd == "backfill":
        backfill(_date(args.start), _date(args.end), args.sites.split(",") if args.sites else None, args.max_articles)
    elif args.cmd == "gdelt-backfill":
        gdelt_backfill(_date(args.start), _date(args.end), fetch_bodies=not args.headlines_only, max_articles=args.max_articles)
    else:
        run_live(args.max_cycles)


if __name__ == "__main__":
    main()
