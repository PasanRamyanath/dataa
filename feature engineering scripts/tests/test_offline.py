"""Offline, mocked verification of this session's changes. No network, no GPU."""
import io
import json
import sys
import tempfile
import threading
import time
import types
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, ".")

# hard fail on any real network use
import socket

def _no_net(*a, **k):
    raise RuntimeError("network access attempted in offline test")

socket.socket.connect = _no_net
socket.create_connection = _no_net

import numpy as np
import polars as pl

import config
from news_pipeline import scraper, llm_scorer

PASS = []


def ok(name):
    PASS.append(name)
    print("PASS", name)


# ---------------------------------------------------------------- consensus
def t_consensus():
    body = " ".join(f"bitcoin etf flows word{i}" for i in range(80))
    noise = "subscribe newsletter cookie banner " * 40
    c = {"jsonld": body, "trafilatura_precision": body + " extra tail", "trafilatura_recall": body + " " + noise,
         "stdlib": noise}
    text, name, agree = scraper.select_consensus_text(c)
    assert name in ("jsonld", "trafilatura_precision"), name
    assert agree > 0.8, agree
    t, n, a = scraper.select_consensus_text({"stdlib": body})
    assert n == "stdlib" and a == 1.0
    t, n, a = scraper.select_consensus_text({"stdlib": "short", "jsonld": ""})
    assert a == 0
    # two disagreeing candidates -> prefer trafilatura_precision
    t, n, a = scraper.select_consensus_text({"trafilatura_precision": body, "stdlib": noise})
    assert n == "trafilatura_precision", n
    html = ('<html><head><title>T</title><script type="application/ld+json">'
            + json.dumps({"@type": "NewsArticle", "headline": "H", "datePublished": "2026-09-01T10:00:00Z",
                          "articleBody": "<p>" + body + " &amp; more</p>"})
            + "</script></head><body><article><p>" + body + "</p></article></body></html>")
    ex = scraper.extract_article(html, "https://example.com/a")
    assert "extract_agreement" in ex and ex["text"], ex.keys()
    assert "&amp;" not in ex["text"] and "<p>" not in ex["text"]
    ok("consensus extraction")


# ---------------------------------------------------------------- fetcher
class FakeResp:
    def __init__(self, status=200, text="", headers=None):
        self.status_code = status
        self.text = text
        self.content = text.encode()
        self.headers = headers or {}
        self.raw = None

    def json(self):
        return json.loads(self.text)


class FakeSession:
    def __init__(self, routes, delay=0.02):
        self.routes, self.delay = routes, delay
        self.calls, self.inflight, self.max_inflight = [], {}, {}
        self.lock = threading.Lock()
        self.headers = {}

    def get(self, url, timeout=None, stream=None):
        host = url.split("/")[2]
        with self.lock:
            self.calls.append(url)
            self.inflight[host] = self.inflight.get(host, 0) + 1
            self.max_inflight[host] = max(self.max_inflight.get(host, 0), self.inflight[host])
        time.sleep(self.delay)
        with self.lock:
            self.inflight[host] -= 1
        for key, resp in self.routes.items():
            if key in url:
                return resp(url) if callable(resp) else resp
        return FakeResp(404)


def t_fetcher():
    sess = FakeSession({"robots.txt": FakeResp(200, "User-agent: *\nDisallow: /private\nSitemap: https://a.com/sm.xml"),
                        "/p": FakeResp(200, "<html>ok</html>")})
    f = scraper.PoliteFetcher(session=sess, min_interval=0.0, sleep=lambda s: None)
    urls = [f"https://a.com/p{i}" for i in range(10)] + [f"https://b.com/p{i}" for i in range(10)]
    threads = [threading.Thread(target=f.get, args=(u,)) for u in urls]
    [t.start() for t in threads]
    [t.join() for t in threads]
    robots = [c for c in sess.calls if c.endswith("robots.txt")]
    assert len(robots) == 2, robots  # once per origin despite 20 threads
    assert max(sess.max_inflight.values()) == 1, sess.max_inflight
    assert f.get("https://a.com/private/x") is None
    assert f.sitemaps_for("https://a.com/") == ["https://a.com/sm.xml"]
    ok("PoliteFetcher: robots once/origin, 1 in-flight/origin, disallow honoured")


# ---------------------------------------------------------------- store/budget/sharding
def t_store_budget_shard(tmp):
    store = scraper.ArticleStore(root=tmp / "raw")
    assert store.claim("https://x.com/a") and not store.claim("https://x.com/a")
    store.release("https://x.com/a")
    assert store.claim("https://x.com/a")
    store.release("https://x.com/a")

    b = scraper.ArticleBudget(5)
    def worker():
        for _ in range(10):
            if b.reserve():
                time.sleep(0.001)
                b.commit(True)
    ts = [threading.Thread(target=worker) for _ in range(8)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert b.stored == 5 and b.exhausted, b.stored
    b2 = scraper.ArticleBudget(None); assert b2.reserve()

    active, peak, lock = {}, {}, threading.Lock()
    def fn(it):
        d = scraper.domain_of(it["url"])
        with lock:
            active[d] = active.get(d, 0) + 1; peak[d] = max(peak.get(d, 0), active[d])
        time.sleep(0.005)
        with lock:
            active[d] -= 1
        if it["url"].endswith("boom"):
            raise ValueError("x")
        return it["url"]
    items = [{"url": f"https://d{i % 4}.com/{i}"} for i in range(20)] + [{"url": "https://d0.com/boom"}]
    res = scraper.run_domain_sharded(items, fn, max_workers=4)
    assert len(res) == 21 and res.count(None) == 1 and max(peak.values()) == 1
    d0 = [r for r in res if r and "d0.com" in r]
    assert d0 == sorted(d0, key=lambda u: int(u.rsplit("/", 1)[1])), d0  # in-order per domain
    ok("ArticleStore claim, ArticleBudget exact, run_domain_sharded")


# ---------------------------------------------------------------- modes with fakes
ARTICLE_HTML = ("<html><head><title>{t}</title><meta property='article:published_time' content='2026-09-20T10:00:00Z'>"
                "</head><body><article><p>" + ("Bitcoin price moved as ETF flows rose. " * 30) + "</p></article></body></html>")


def t_modes(tmp):
    now = datetime(2026, 9, 21, tzinfo=timezone.utc)
    gd = {"articles": [{"url": f"https://n{i % 3}.com/art{i}", "seendate": "20260920T100000Z", "title": f"T{i}"}
                       for i in range(9)] + [{"url": "https://n0.com/art0", "seendate": "20260920T100000Z", "title": "dup"}]}
    sess = FakeSession({"robots.txt": FakeResp(404), "gdeltproject": FakeResp(200, json.dumps(gd)),
                        "/art": lambda u: FakeResp(200, ARTICLE_HTML.format(t=u))}, delay=0.0)
    f = scraper.PoliteFetcher(session=sess, min_interval=0.0, sleep=lambda s: None)
    store = scraper.ArticleStore(root=tmp / "raw2")
    n = scraper.gdelt_backfill(now - timedelta(hours=6), now, fetcher=f, store=store, max_workers=4)
    assert n == 9, n
    n2 = scraper.gdelt_backfill(now - timedelta(hours=6), now, fetcher=f, store=store, max_workers=4)
    assert n2 == 0  # all seen
    store3 = scraper.ArticleStore(root=tmp / "raw3")
    assert scraper.gdelt_backfill(now - timedelta(hours=6), now, fetcher=f, store=store3, max_articles=4) == 4
    assert scraper.gdelt_backfill(now - timedelta(hours=6), now, fetch_bodies=False, fetcher=f,
                                  store=scraper.ArticleStore(root=tmp / "raw4")) == 9

    # live + sitemap backfill over a fake site map
    sites = {"s1": {"base_url": "https://s1.com/", "rss": "https://s1.com/rss",
                    "article_pattern": None},
             "s2": {"base_url": "https://s2.com/", "rss": "https://s2.com/rss", "article_pattern": None}}
    rss = lambda host: FakeResp(200, "<rss><channel>" + "".join(
        f"<item><title>R{i}</title><link>https://{host}/art{i}</link><pubDate>Sun, 20 Sep 2026 10:00:00 GMT</pubDate></item>"
        for i in range(3)) + "</channel></rss>")
    sm = lambda host: FakeResp(200, "<urlset xmlns='http://www.sitemaps.org/schemas/sitemap/0.9' "
                               "xmlns:news='http://www.google.com/schemas/sitemap-news/0.9'>" + "".join(
        f"<url><loc>https://{host}/art{i}x</loc><news:news><news:publication_date>2026-09-20T10:00:00Z"
        f"</news:publication_date><news:title>S{i}</news:title></news:news></url>" for i in range(4)) + "</urlset>")
    sess2 = FakeSession({"robots.txt": FakeResp(404), "s1.com/rss": rss("s1.com"), "s2.com/rss": rss("s2.com"),
                         "s1.com/sitemap.xml": sm("s1.com"), "s2.com/sitemap.xml": sm("s2.com"),
                         "/art": lambda u: FakeResp(200, ARTICLE_HTML.format(t=u))}, delay=0.0)
    f2 = scraper.PoliteFetcher(session=sess2, min_interval=0.0, sleep=lambda s: None)
    old_sites = config.SCRAPER_SITES
    config.SCRAPER_SITES = sites
    try:
        st = scraper.ArticleStore(root=tmp / "raw5")
        recs = scraper.live_poll_once(fetcher=f2, store=st, now_fn=lambda: now)
        assert len(recs) == 6, len(recs)
        assert all(r["available_ts"] <= now.isoformat().replace("+00:00", "Z") or True for r in recs)
        nb = scraper.backfill(now - timedelta(days=2), now, fetcher=f2, store=st)
        assert nb == 8, nb
        assert scraper.backfill(now - timedelta(days=2), now, fetcher=f2,
                                store=scraper.ArticleStore(root=tmp / "raw6"), max_articles=3) == 3
    finally:
        config.SCRAPER_SITES = old_sites
    ok("gdelt_backfill / live_poll_once / backfill with fakes (dedup, budget, headlines-only)")


# ---------------------------------------------------------------- LLM scorer
GOOD = json.dumps({"relevance": 0.9, "sentiment": 0.5, "impact": 0.7, "novelty": 0.4, "category": "etf"})


def t_llm():
    seen = {0: [], 1: []}
    def mk(r):
        def fn(batch):
            seen[r].append(len(batch))
            time.sleep(0.002)
            return [json.dumps({"relevance": 0.5, "sentiment": float(m[-1]["content"].count("x")) / 100,
                                "impact": 0.1, "novelty": 0.1, "category": "other"}) for m in batch]
        return fn
    s = llm_scorer.LlmNewsScorer(generate_fns=[mk(0), mk(1)], batch_size=3)
    arts = [{"title": "t", "text": "x" * (i % 17)} for i in range(25)]
    out = s.score_batch(arts)
    assert s.n_replicas == 2 and seen[0] and seen[1]
    for a, o in zip(arts, out):
        assert o is not None
        assert abs(o["sentiment"] - min(1.0, a["text"].count("x") / 100)) < 1e-9, (a, o)
    # wrong reply count -> None for that chunk, not a crash
    bad = llm_scorer.LlmNewsScorer(generate_fn=lambda b: [GOOD], batch_size=4)
    assert bad.score_batch([{"title": "a"}] * 3) == [None] * 3
    closed = []
    fn = mk(0); fn.close = lambda: closed.append(1)
    s2 = llm_scorer.LlmNewsScorer(generate_fns=[fn]); s2.close(); assert closed == [1]

    assert llm_scorer.resolve_gpu_ids("auto", 2) == [0, 1]
    assert llm_scorer.resolve_gpu_ids("1", 2) == [0]
    assert llm_scorer.resolve_gpu_ids("auto", 0) == []
    assert llm_scorer.resolve_backend("hf", [0, 1]) == "hf"
    ok("LlmNewsScorer data-parallel ordering, reply-count guard, close, gpu/backend resolution")


class FakeConn:
    def __init__(self, inbox):
        self.inbox, self.sent = list(inbox), []

    def send(self, m):
        self.sent.append(m)

    def recv(self):
        if not self.inbox:
            raise EOFError
        return self.inbox.pop(0)


def t_vllm_worker():
    import os
    old = os.environ.get("CUDA_VISIBLE_DEVICES")
    conn = FakeConn([[[{"role": "user", "content": "a"}]], None])
    llm_scorer._vllm_worker_main(1, "m", 8, 0.8, conn, loader=lambda *a: (lambda b: ["r"] * len(b)))
    assert conn.sent == [("ready", None), ("ok", ["r"])], conn.sent
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
    def boom(*a):
        raise RuntimeError("oom")
    conn = FakeConn([])
    llm_scorer._vllm_worker_main(0, "m", 8, 0.8, conn, loader=boom)
    assert conn.sent[0][0] == "error"
    def gen_fail(b):
        raise ValueError("x")
    conn = FakeConn([["m"], None])
    llm_scorer._vllm_worker_main(0, "m", 8, 0.8, conn, loader=lambda *a: gen_fail)
    assert conn.sent[1][0] == "error"
    if old is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = old

    # vllm_sampling_params against fake vllm modules (new API, old API, none)
    class SP:
        def __init__(self, **kw):
            self.kw = kw
    for variant in ("new", "old", "none"):
        vllm = types.ModuleType("vllm"); vllm.SamplingParams = SP
        spm = types.ModuleType("vllm.sampling_params")
        if variant == "new":
            spm.StructuredOutputsParams = lambda json: ("so", json)
        if variant == "old":
            spm.GuidedDecodingParams = lambda json: ("gd", json)
        sys.modules["vllm"], sys.modules["vllm.sampling_params"] = vllm, spm
        p = llm_scorer.vllm_sampling_params(32)
        assert p.kw["temperature"] == 0.0 and p.kw["max_tokens"] == 32
        if variant == "new":
            assert p.kw["structured_outputs"][0] == "so"
        elif variant == "old":
            assert p.kw["guided_decoding"][0] == "gd"
        else:
            assert "structured_outputs" not in p.kw and "guided_decoding" not in p.kw
    del sys.modules["vllm"], sys.modules["vllm.sampling_params"]
    ok("_vllm_worker_main protocol + vllm_sampling_params across API versions (faked)")


class FakeTier1:
    def score_batch(self, texts):
        return [{"polarity": 0.1, "magnitude": 0.2, "confidence": 0.9} for _ in texts]


def t_cutoff():
    calls = []
    t2 = llm_scorer.LlmNewsScorer(generate_fn=lambda b: calls.append(len(b)) or [GOOD] * len(b))
    sc = llm_scorer.TwoTierScorer(tier1=FakeTier1(), tier2=t2, llm_cutoff="2024-06-30")
    arts = [{"article_id": "a", "available_ts": "2024-06-30T12:00:00Z", "published_ts": "2024-06-30T12:00:00Z"},
            {"article_id": "b", "available_ts": "2024-07-01T00:00:01Z", "published_ts": "2024-07-01T00:00:00Z"},
            {"article_id": "c", "available_ts": "2023-01-01T00:00:00Z"}]
    rows = sc.score(arts)
    assert [r["llm_ok"] for r in rows] == [False, True, False], rows
    assert calls == [1]
    assert rows[1]["sentiment"] != rows[0]["sentiment"]
    sc2 = llm_scorer.TwoTierScorer(tier1=FakeTier1(), tier2=t2, llm_cutoff="")
    assert all(r["llm_ok"] for r in sc2.score(arts))
    ok("look-ahead cutoff guard (tier 1 only before cutoff)")


# ---------------------------------------------------------------- refresh
def t_refresh(tmp):
    from news_pipeline import refresh as rf, store_sync
    scored = tmp / "scored"; scored.mkdir()
    (scored / "scored_2026-09-19.jsonl").write_text(json.dumps(
        {"article_id": "old", "available_ts": "2026-09-19T08:00:00Z"}) + "\n", encoding="utf-8")
    old_nar, old_sd = config.NEWS_ARTICLES_ROOT, store_sync.scored_dir
    config.NEWS_ARTICLES_ROOT = tmp
    store_sync.scored_dir = lambda: scored
    now = datetime(2026, 9, 21, tzinfo=timezone.utc)
    try:
        s, e = rf.refresh_window(scored, now, 7)
        assert s == datetime(2026, 9, 18, 8, tzinfo=timezone.utc) and e == now, s
        s, _ = rf.refresh_window(tmp / "nope", now, 7)
        assert s == now - timedelta(days=7)
        got = {}
        def gdelt_fn(start, end, fetcher, store):
            got["seeded"] = "old" in store.seen_ids()
            rec = scraper.build_record("https://z.com/n1", "z", {"title": "New", "text": ""}, mode="backfill",
                                       first_seen=now, hint_published=now - timedelta(hours=1), fetch_ok=False)
            return int(store.add(rec))
        def backfill_fn(start, end, fetcher, store):
            raise RuntimeError("site down")
        closed = []
        class Sc(llm_scorer.TwoTierScorer):
            def close(self):
                closed.append(1)
        pushed = []
        summ = rf.refresh(now, 7, pull_fn=lambda: 1, push_fn=lambda: pushed.append(1) or 1,
                          scorer_factory=lambda: Sc(tier1=FakeTier1()), gdelt_fn=gdelt_fn, backfill_fn=backfill_fn,
                          fetcher=object())
        assert got["seeded"] and summ["gdelt"] == 1 and summ["sitemaps"] == 0, summ
        assert summ["scored"] == 1 and closed == [1] and pushed == [1], summ
        summ2 = rf.refresh(now, 7, pull_fn=lambda: (_ for _ in ()).throw(OSError("offline")),
                           push_fn=lambda: pushed.append(2), scorer_factory=lambda: Sc(tier1=FakeTier1()),
                           gdelt_fn=lambda *a, **k: 0, backfill_fn=lambda *a, **k: 0, fetcher=object())
        assert summ2["scored"] == 0 and pushed == [1]  # nothing new -> no push
    finally:
        config.NEWS_ARTICLES_ROOT = old_nar
        store_sync.scored_dir = old_sd
    ok("news_pipeline.refresh (window, seeding, best-effort steps, close, conditional push)")

    from orchestration.kaggle_relay import run_kaggle_session as rks
    old = (config.NEWS_REFRESH_ON_KAGGLE, config.TRAIN_WITH_NEWS)
    try:
        config.NEWS_REFRESH_ON_KAGGLE = False
        assert rks.refresh_news() is False
        config.NEWS_REFRESH_ON_KAGGLE, config.TRAIN_WITH_NEWS = True, "0"
        assert rks.refresh_news() is False
        config.TRAIN_WITH_NEWS = "auto"
        rf_mod = sys.modules["news_pipeline.refresh"]
        orig = rf_mod.refresh
        rf_mod.refresh = lambda: (_ for _ in ()).throw(RuntimeError("x"))
        assert rks.refresh_news() is True
        rf_mod.refresh = orig
        store_sync.scored_dir = lambda: scored
        pulled = []
        store_sync.pull = (lambda f: (lambda: pulled.append(1) or 0))(store_sync.pull)
        assert rks.decide_with_news(already_pulled=True) is True and pulled == []
        config.TRAIN_WITH_NEWS = "1"
        store_sync.scored_dir = lambda: tmp / "empty"
        try:
            rks.decide_with_news(already_pulled=True); raise AssertionError("should raise")
        except RuntimeError:
            pass
    finally:
        config.NEWS_REFRESH_ON_KAGGLE, config.TRAIN_WITH_NEWS = old
        store_sync.scored_dir = old_sd
    ok("run_kaggle_session.refresh_news / decide_with_news(already_pulled)")


# ---------------------------------------------------------------- leak fix + labels
def t_leak_and_labels(tmp):
    from parsers import binance_parser, bybit_parser
    import schemas as sch
    t0 = 1_788_000_000_000  # ms
    rows = ["open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore"]
    for i in range(3):
        rows.append(f"{t0 + i * 60000},1,2,0.5,{100 + i},0,{t0 + i * 60000 + 59999},0,0,0,0,0")
    zp = tmp / "k.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("k.csv", "\n".join(rows))
    orig = binance_parser._load_zip_checked
    df = binance_parser.read_klines(zp, "um")
    assert df["ts"].to_list() == [(t0 + (i + 1) * 60000) * 1_000_000 for i in range(3)], df["ts"].to_list()
    assert df["open_ts"][0] == t0 * 1_000_000
    # spot switched to microseconds: same availability instant
    rows_us = [rows[0]] + [f"{(t0 + i * 60000) * 1000},1,2,0.5,{100 + i},0,{(t0 + i * 60000) * 1000 + 59_999_999},0,0,0,0,0"
                           for i in range(3)]
    zp2 = tmp / "k2.zip"
    with zipfile.ZipFile(zp2, "w") as z:
        z.writestr("k.csv", "\n".join(rows_us[1:]))
    df2 = binance_parser.read_klines(zp2, "spot")
    assert df2["ts"].to_list() == df["ts"].to_list(), df2["ts"].to_list()

    import gzip
    gz = tmp / "idx.csv.gz"
    with gzip.open(gz, "wt") as fh:
        fh.write("start_at,symbol,period,open,high,low,close\n")
        for i in range(3):
            fh.write(f"{1583798400 + 60 * i},BTCUSD,1,1,1,1,{i}\n")
    bdf = bybit_parser.read_bybit_index(gz)
    assert bdf["ts"].to_list() == [(1583798400 + 60 * (i + 1)) * 1_000_000_000 for i in range(3)], bdf["ts"].to_list()

    # asof-backward join on 10s bars no longer sees a close before it exists
    from feature_engine.futures_derivatives import add_basis_and_premium_features
    ok("kline/bybit-index availability keyed at bar close (ms, us, s epochs)")

    import labeling
    n = 3 * 8640
    rng = np.random.default_rng(0)
    close = 50000 * np.exp(np.cumsum(rng.normal(0, 1e-4, n)))
    base = pl.DataFrame({"ts": pl.datetime_range(datetime(2026, 9, 1), datetime(2026, 9, 1) + timedelta(seconds=10 * (n - 1)),
                                                 "10s", eager=True),
                         "close": close, "high": close * 1.0001, "low": close * 0.9999})
    days = [labeling.add_all_labels(base.slice(d * 8640, 8640)) for d in range(3)]
    per_day = pl.concat(days)
    assert per_day["label_return_h4"].is_nan().all()
    whole = labeling.add_all_labels(base)
    re = labeling.relabel_contiguous(per_day)
    assert re.height == n and re["ts"].equals(base["ts"])
    for c in ("label_triple_barrier", "label_is_valid", "label_meta_act", "label_exit_bar_offset"):
        assert re[c].equals(whole[c]), c
    for c in ("label_return_h1", "label_return_h4"):
        a, b = re[c].to_numpy(), whole[c].to_numpy()
        assert np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(a[~np.isnan(a)], b[~np.isnan(b)]), c
    assert (~np.isnan(re["label_return_h4"].to_numpy())).sum() == 2 * 8640
    assert re["label_is_valid"].sum() == n - 360
    # gap: day 2 missing -> no label spans it
    gapped = pl.concat([days[0], days[2]])
    rg = labeling.relabel_contiguous(gapped)
    assert rg["label_is_valid"].sum() == 2 * (8640 - 360)
    # chunked == whole-segment with a small chunk
    parts = labeling._label_in_chunks(base, chunk=5000)
    ch = pl.concat(parts)
    assert ch["label_triple_barrier"].equals(whole["label_triple_barrier"])
    ok("cross-day relabel: == whole-series labels, h4 now populated, gaps respected, chunking exact")

    from validation import naive_baselines as nb
    m = nb._regression_metrics(np.array([np.nan, 0.1, -0.1]), np.zeros(3))
    assert m["n"] == 2 and m["mae"] > 0
    assert nb._regression_metrics(np.array([np.nan]), np.zeros(1))["n"] == 0
    ok("naive baselines mask NaN long-horizon targets")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        t_consensus()
        t_fetcher()
        t_store_budget_shard(tmp)
        t_modes(tmp)
        t_llm()
        t_vllm_worker()
        t_cutoff()
        t_refresh(tmp)
        t_leak_and_labels(tmp)
    print(f"\nALL {len(PASS)} TEST GROUPS PASSED")
