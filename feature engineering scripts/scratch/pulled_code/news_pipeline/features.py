"""
Point-in-time, multi-window news features for the 10s bars (plan §5.1 /
§9.3): NEWS_FEATURE_COLUMNS and attach_news_features(df).

Kept OUTSIDE common/feature_contract.py on purpose: FEATURE_COLUMNS mirrors
the feature-engineering parquet schema (226-column assert), while news
features come from a separate store and are joined on demand. Consumers opt
in (models/sentiment_branch --with-news, models/gbm_family --with-news) and
record the exact column list in their checkpoint's metrics.json, which the
live predictors read to decide whether to attach news at inference.

Leakage rules this module enforces:
- Only articles with available_ts <= bar ts are visible to that bar
  (available_ts is first-seen for live articles, published + lag for
  backfilled ones — see news_pipeline.scraper). Bar ts is the bar OPEN
  (resample.py labels bars left), so this is conservative by one bar.
- Near-duplicate stories are clustered strictly forward in time: an article
  is a duplicate only of a story that was ALREADY available, so the first
  outlet's timestamp wins and later reposts count as "spread", never as new
  stories.
- Windows are exact differences of prefix sums over the event stream
  (searchsorted on available_ts) — each article is counted once per window
  it falls in, never forward-filled forever (the addendum's asof-join
  pitfall). Empty/missing store → the same columns, all zero, with
  news_coverage = 0 (identical schema on every path).

Features (all Float32):
  per window w in 15m/1h/4h/24h, over unique stories:
    news_n_{w}          story count with relevance >= 0.3 (mention velocity)
    news_sent_{w}       weight-averaged sentiment
    news_pressure_{w}   sum(weight * sentiment)  (signed, size-aware)
    news_disp_{w}       weighted sentiment dispersion (disagreement)
  news_dup_1h              reposts of already-seen stories (story spreading)
  news_n_surprise_1h       news_n_1h - news_n_24h / 24
  news_sent_surprise_1h    news_sent_1h - news_sent_24h
  news_decay_30m/_6h       exponentially decayed pressure (half-lives 30m/6h)
  news_mins_since_major    minutes since last story with weight >= 0.5 (cap 1440)
  news_last_major_sent     that story's sentiment
  news_cat_{c}_4h          4h pressure per event category
  news_llm_share_24h       share of 24h stories scored by the tier-2 LLM
  news_coverage            1 inside the store's covered time range, else 0

weight = relevance * (0.25 + 0.75 * impact) * (0.5 + 0.5 * novelty)
"""
from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

import config

log = logging.getLogger("news_pipeline.features")

WINDOWS = {"15m": 900, "1h": 3600, "4h": 4 * 3600, "24h": 24 * 3600}
DECAY_HALF_LIVES = {"30m": 1800, "6h": 6 * 3600}
FEATURE_CATEGORIES = ("regulation", "hack_exploit", "etf_institutional", "macro", "adoption_partnership")
RELEVANCE_COUNT_THRESHOLD = 0.3
MAJOR_WEIGHT_THRESHOLD = 0.5
COVERAGE_TAIL_SECONDS = 6 * 3600  # bars up to this long after the newest article still count as covered
_NS = 1_000_000_000

NEWS_FEATURE_COLUMNS = (
    [f"news_{kind}_{w}" for w in WINDOWS for kind in ("n", "sent", "pressure", "disp")]
    + ["news_dup_1h", "news_n_surprise_1h", "news_sent_surprise_1h"]
    + [f"news_decay_{h}" for h in DECAY_HALF_LIVES]
    + ["news_mins_since_major", "news_last_major_sent"]
    + [f"news_cat_{c}_4h" for c in FEATURE_CATEGORIES]
    + ["news_llm_share_24h", "news_coverage"]
)

_STOP = {"the", "and", "for", "with", "that", "this", "from", "into", "after", "over", "amid", "says", "said",
         "will", "are", "was", "its", "has", "have", "new", "what", "why", "how", "you", "your", "not", "but"}


def _title_tokens(title: str) -> frozenset:
    return frozenset(t for t in re.findall(r"[a-z0-9$]+", (title or "").lower()) if len(t) >= 3 and t not in _STOP)


def assign_stories(titles: list[str], times_ns: np.ndarray,
                   threshold: float = config.SCRAPER_DEDUP_JACCARD,
                   window_s: float = config.SCRAPER_DEDUP_WINDOW_HOURS * 3600) -> np.ndarray:
    """Returns is_first (bool array) for articles already sorted by
    available_ts: False when the title's token-set Jaccard with an earlier
    story first seen within `window_s` is >= threshold. Inverted index on
    tokens keeps this ~linear."""
    n = len(titles)
    is_first = np.ones(n, dtype=bool)
    story_tokens: list[frozenset] = []
    story_time: list[int] = []
    index: dict[str, list[int]] = {}
    window_ns = int(window_s * _NS)
    for i in range(n):
        toks = _title_tokens(titles[i])
        t = int(times_ns[i])
        match = False
        if toks:
            cand: set[int] = set()
            for tok in toks:
                cand.update(index.get(tok, ()))
            for sid in cand:
                if t - story_time[sid] > window_ns:
                    continue
                other = story_tokens[sid]
                inter = len(toks & other)
                if inter and inter / len(toks | other) >= threshold:
                    match = True
                    break
        if match:
            is_first[i] = False
            continue
        sid = len(story_tokens)
        story_tokens.append(toks)
        story_time.append(t)
        for tok in toks:
            index.setdefault(tok, []).append(sid)
    return is_first


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
_SCHEMA = {"available_ts": pl.Utf8, "title": pl.Utf8, "sentiment": pl.Float64, "relevance": pl.Float64,
           "impact": pl.Float64, "novelty": pl.Float64, "category": pl.Utf8, "llm_ok": pl.Boolean}
_cache: dict = {"sig": None, "df": None, "arrays_for": None, "arrays": None}


def _empty_articles() -> pl.DataFrame:
    return pl.DataFrame(schema=_SCHEMA)


def load_scored_articles(scored_root: Path | None = None) -> pl.DataFrame:
    """All scored articles, sorted by available_ts, cached until any
    scored_*.jsonl file changes size (cheap to call every live tick)."""
    scored_root = Path(scored_root or config.NEWS_ARTICLES_ROOT / "scored")
    files = sorted(scored_root.glob("scored_*.jsonl")) if scored_root.exists() else []
    sig = (str(scored_root), tuple((p.name, p.stat().st_size) for p in files))
    if _cache["sig"] == sig:
        return _cache["df"]
    rows = []
    for path in files:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows.append({k: r.get(k) for k in _SCHEMA})
    df = pl.DataFrame(rows, schema=_SCHEMA) if rows else _empty_articles()
    df = df.drop_nulls(["available_ts", "sentiment"]).sort("available_ts")
    _cache["sig"], _cache["df"] = sig, df
    return df


_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _iso_to_ns(s: str) -> int:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    dt = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return (dt - _EPOCH) // timedelta(microseconds=1) * 1000


def _articles_to_arrays(articles: pl.DataFrame) -> dict | None:
    if articles.height == 0:
        return None
    ts = np.array([_iso_to_ns(s) for s in articles["available_ts"].to_list()], dtype=np.int64)
    order = np.argsort(ts, kind="stable")
    ts = ts[order]
    get = lambda c, fill: articles[c].fill_null(fill).to_numpy()[order]  # noqa: E731
    sent = get("sentiment", 0.0).astype(np.float64)
    rel = get("relevance", 0.0).astype(np.float64)
    impact = get("impact", 0.5).astype(np.float64)
    novelty = get("novelty", 0.5).astype(np.float64)
    weight = rel * (0.25 + 0.75 * impact) * (0.5 + 0.5 * novelty)
    titles = [articles["title"][int(i)] or "" for i in order]
    return {"ts": ts, "sent": sent, "rel": rel, "weight": weight,
            "category": articles["category"].fill_null("other").to_numpy()[order],
            "llm": articles["llm_ok"].fill_null(False).to_numpy()[order].astype(np.float64),
            "is_first": assign_stories(titles, ts)}


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------
def _window_sum(ts: np.ndarray, values: np.ndarray, T: np.ndarray, hi: np.ndarray, window_s: int) -> np.ndarray:
    prefix = np.concatenate([[0.0], np.cumsum(values, dtype=np.float64)])
    lo = np.searchsorted(ts, T - window_s * _NS, side="right")
    return prefix[hi] - prefix[lo]


def _decayed(ts: np.ndarray, values: np.ndarray, T: np.ndarray, hi: np.ndarray, half_life_s: float) -> np.ndarray:
    tau = half_life_s / math.log(2) * _NS
    D = np.empty(len(ts), dtype=np.float64)
    acc, prev = 0.0, None
    for k in range(len(ts)):
        if prev is not None:
            acc *= math.exp(-(ts[k] - prev) / tau)
        acc += values[k]
        D[k] = acc
        prev = ts[k]
    idx = hi - 1
    out = np.zeros(len(T), dtype=np.float64)
    ok = idx >= 0
    out[ok] = D[idx[ok]] * np.exp(-(T[ok] - ts[idx[ok]]) / tau)
    return out


def compute_news_features(bar_ts_ns: np.ndarray, articles: pl.DataFrame | None = None,
                          asof_offset_seconds: int = 0) -> dict[str, np.ndarray]:
    """{column: float32 array aligned to bar_ts_ns (int64 ns, UTC-naive)}."""
    if articles is None:
        # Live path (every tick): reuse the store's arrays — the near-dupe
        # clustering in _articles_to_arrays is a Python loop over the whole
        # store, only redone when a scored_*.jsonl file changes.
        articles = load_scored_articles()
        if _cache["arrays_for"] is not articles:
            _cache["arrays"], _cache["arrays_for"] = _articles_to_arrays(articles), articles
        arr = _cache["arrays"]
    else:
        arr = _articles_to_arrays(articles)
    T = np.asarray(bar_ts_ns, dtype=np.int64) + asof_offset_seconds * _NS
    out = {c: np.zeros(len(T), dtype=np.float32) for c in NEWS_FEATURE_COLUMNS}
    if arr is None or len(T) == 0:
        return out

    covered = (T >= arr["ts"][0]) & (T <= arr["ts"][-1] + COVERAGE_TAIL_SECONDS * _NS)
    first = arr["is_first"]
    ts_all = arr["ts"]
    ts_s = ts_all[first]
    w, s, rel = arr["weight"][first], arr["sent"][first], arr["rel"][first]
    cat, llm = arr["category"][first], arr["llm"][first]
    hi = np.searchsorted(ts_s, T, side="right")

    means = {}
    for name, secs in WINDOWS.items():
        n = _window_sum(ts_s, (rel >= RELEVANCE_COUNT_THRESHOLD).astype(np.float64), T, hi, secs)
        sw = _window_sum(ts_s, w, T, hi, secs)
        sws = _window_sum(ts_s, w * s, T, hi, secs)
        sws2 = _window_sum(ts_s, w * s * s, T, hi, secs)
        has = sw > 1e-9
        mean = np.where(has, sws / np.where(has, sw, 1.0), 0.0)
        var = np.where(has, sws2 / np.where(has, sw, 1.0) - mean ** 2, 0.0)
        means[name] = mean
        out[f"news_n_{name}"] = n
        out[f"news_sent_{name}"] = mean
        out[f"news_pressure_{name}"] = sws
        out[f"news_disp_{name}"] = np.sqrt(np.clip(var, 0.0, None))

    hi_all = np.searchsorted(ts_all, T, side="right")
    out["news_dup_1h"] = _window_sum(ts_all, (~first).astype(np.float64), T, hi_all, 3600)
    out["news_n_surprise_1h"] = out["news_n_1h"] - out["news_n_24h"] / 24.0
    out["news_sent_surprise_1h"] = means["1h"] - means["24h"]
    for name, hl in DECAY_HALF_LIVES.items():
        out[f"news_decay_{name}"] = _decayed(ts_s, w * s, T, hi, hl)

    major = w >= MAJOR_WEIGHT_THRESHOLD
    ts_m, s_m = ts_s[major], s[major]
    hm = np.searchsorted(ts_m, T, side="right") - 1
    has_m = hm >= 0
    mins = np.full(len(T), 1440.0)
    mins[has_m] = np.minimum(1440.0, (T[has_m] - ts_m[hm[has_m]]) / (60 * _NS))
    last_sent = np.zeros(len(T))
    last_sent[has_m] = np.where(mins[has_m] < 1440.0, s_m[hm[has_m]], 0.0)
    out["news_mins_since_major"] = mins
    out["news_last_major_sent"] = last_sent

    for c in FEATURE_CATEGORIES:
        out[f"news_cat_{c}_4h"] = _window_sum(ts_s, np.where(cat == c, w * s, 0.0), T, hi, WINDOWS["4h"])
    n24 = _window_sum(ts_s, np.ones(len(ts_s)), T, hi, WINDOWS["24h"])
    out["news_llm_share_24h"] = np.where(n24 > 0, _window_sum(ts_s, llm, T, hi, WINDOWS["24h"]) / np.maximum(n24, 1), 0.0)

    for c in NEWS_FEATURE_COLUMNS:
        if c == "news_mins_since_major":
            out[c] = np.where(covered, out[c], 1440.0).astype(np.float32)
        else:
            out[c] = np.where(covered, out[c], 0.0).astype(np.float32)
    out["news_coverage"] = covered.astype(np.float32)
    return out


def bar_ts_to_ns(ts: pl.Series) -> np.ndarray:
    if isinstance(ts.dtype, pl.Datetime) and ts.dtype.time_zone:
        ts = ts.dt.convert_time_zone("UTC").dt.replace_time_zone(None)
    return ts.dt.cast_time_unit("ns").cast(pl.Int64).to_numpy()


def attach_news_features(df: pl.DataFrame, articles: pl.DataFrame | None = None,
                         ts_col: str = "ts") -> pl.DataFrame:
    """Returns df with NEWS_FEATURE_COLUMNS added (replacing any existing
    ones). Row order and count are unchanged."""
    feats = compute_news_features(bar_ts_to_ns(df[ts_col]), articles)
    return df.drop([c for c in NEWS_FEATURE_COLUMNS if c in df.columns]).with_columns(
        [pl.Series(c, feats[c], dtype=pl.Float32) for c in NEWS_FEATURE_COLUMNS]
    )


def store_summary(articles: pl.DataFrame | None = None) -> dict:
    articles = load_scored_articles() if articles is None else articles
    if articles.height == 0:
        return {"n_articles": 0}
    return {"n_articles": articles.height,
            "first_available_ts": articles["available_ts"].min(),
            "last_available_ts": articles["available_ts"].max(),
            "llm_scored_share": float(articles["llm_ok"].fill_null(False).mean())}
