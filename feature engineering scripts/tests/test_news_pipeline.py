"""Offline test of news_pipeline + integrations. No network, no model downloads."""
import gzip
import json
import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(r"F:\Downloads\Binance data\feature engineering scripts")
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="newspipe_"))
config.NEWS_ARTICLES_ROOT = TMP / "news_articles"

from news_pipeline import features as nf, llm_scorer as ls, scraper as sc  # noqa: E402

UTC = timezone.utc


# ---------------------------------------------------------------- fake HTTP
class FakeResp:
    def __init__(self, status, content=b"", headers=None):
        self.status_code, self.content, self.headers, self.raw = status, content, headers or {}, None

    @property
    def text(self):
        return self.content.decode("utf-8")

    def json(self):
        return json.loads(self.content)


class FakeSession:
    def __init__(self, routes):
        self.routes, self.headers, self.calls = routes, {}, []

    def get(self, url, timeout=None, stream=False):
        self.calls.append(url)
        r = self.routes.get(url)
        if r is None:
            return FakeResp(404)
        if isinstance(r, list):  # sequence of responses (retry test)
            return r.pop(0) if len(r) > 1 else r[0]
        return r


def article_html(title, published, body_sentences=12, meta=True):
    meta_tag = f'<meta property="article:published_time" content="{published}">' if meta else ""
    paras = "".join(f"<p>{title} paragraph {i}: Bitcoin ETF inflows and market structure discussed at length here.</p>"
                    for i in range(body_sentences))
    return (f"<html><head><title>{title}</title>{meta_tag}<meta property='og:title' content='{title}'>"
            f"<script>var x = '<p>not text</p>';</script></head><body><nav><p>Menu menu menu menu menu menu menu</p></nav>"
            f"<article>{paras}</article></body></html>").encode()


B = "https://news.example.com"
robots = f"User-agent: *\nDisallow: /private/\nCrawl-delay: 3\nSitemap: {B}/sitemap_index.xml\n".encode()
index_xml = f"""<?xml version="1.0"?><sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
<sitemap><loc>{B}/sitemap-new.xml.gz</loc><lastmod>2026-09-10T00:00:00Z</lastmod></sitemap>
<sitemap><loc>{B}/sitemap-old.xml</loc><lastmod>2020-01-01T00:00:00Z</lastmod></sitemap>
</sitemapindex>""".encode()
urlset_xml = f"""<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
 xmlns:news="http://www.google.com/schemas/sitemap-news/0.9">
<url><loc>{B}/2026/09/05/etf-approved</loc><news:news><news:publication_date>2026-09-05T12:00:00Z</news:publication_date><news:title>ETF approved</news:title></news:news></url>
<url><loc>{B}/2026/09/06/plain-lastmod</loc><lastmod>2026-09-06T08:00:00+00:00</lastmod></url>
<url><loc>{B}/private/2026/09/06/secret</loc><lastmod>2026-09-06T08:00:00Z</lastmod></url>
<url><loc>{B}/tag/bitcoin</loc><lastmod>2026-09-06T08:00:00Z</lastmod></url>
<url><loc>{B}/2026/07/01/too-old</loc><lastmod>2026-07-01T08:00:00Z</lastmod></url>
<url><loc>{B}/2026/09/07/paywalled</loc><news:news><news:publication_date>2026-09-07T09:30:00Z</news:publication_date><news:title>Paywalled scoop</news:title></news:news></url>
</urlset>""".encode()

sleeps = []
routes = {
    f"{B}/robots.txt": FakeResp(200, robots),
    f"{B}/sitemap_index.xml": FakeResp(200, index_xml),
    f"{B}/sitemap-new.xml.gz": FakeResp(200, gzip.compress(urlset_xml)),
    f"{B}/2026/09/05/etf-approved": [FakeResp(503), FakeResp(200, article_html("ETF approved", "2026-09-05T11:59:00Z"))],
    # page meta says 07:55 (precise) -> must win over sitemap lastmod 08:00
    f"{B}/2026/09/06/plain-lastmod": FakeResp(200, article_html("Plain lastmod story", "2026-09-06T07:55:00Z")),
    f"{B}/2026/09/07/paywalled": FakeResp(200, b"<html><head><title>Paywalled scoop</title></head><body><p>Subscribe to read.</p></body></html>"),
}

orig_sites = config.SCRAPER_SITES
config.SCRAPER_SITES = {"example": {"base_url": B, "rss": f"{B}/feed", "article_pattern": r"/20\d\d/\d\d/\d\d/"}}
try:
    session = FakeSession(routes)
    fetcher = sc.PoliteFetcher(session=session, sleep=sleeps.append, min_interval=0.0)
    store = sc.ArticleStore()
    n = sc.backfill(datetime(2026, 8, 1, tzinfo=UTC), datetime(2026, 9, 30, tzinfo=UTC), fetcher=fetcher, store=store)
    recs = {json.loads(l)["url"]: json.loads(l) for p in store.root.glob("*.jsonl") for l in p.read_text(encoding="utf-8").splitlines()}
    assert n == 3, (n, list(recs))
    assert f"{B}/sitemap-old.xml" not in session.calls, "old sub-sitemap should be pruned by lastmod"
    assert not any("/private/" in c for c in session.calls), "robots Disallow ignored"
    assert not any("/tag/" in c for c in session.calls), "article_pattern filter ignored"
    etf = recs[f"{B}/2026/09/05/etf-approved"]
    assert etf["published_ts"].startswith("2026-09-05T12:00:00"), etf  # news-sitemap date wins
    lag = config.NEWS_BACKFILL_AVAILABILITY_LAG_SECONDS
    assert sc.parse_dt(etf["available_ts"]) == sc.parse_dt(etf["published_ts"]) + timedelta(seconds=lag)
    assert etf["body_available"] and "paragraph 3" in etf["text"] and "Menu" not in etf["text"] and "not text" not in etf["text"]
    plain = recs[f"{B}/2026/09/06/plain-lastmod"]
    assert plain["published_ts"].startswith("2026-09-06T07:55:00") and plain["published_source"] == "meta:article:published_time", plain
    pw = recs[f"{B}/2026/09/07/paywalled"]
    assert not pw["body_available"] and pw["text"] == "" and pw["title"] == "Paywalled scoop"
    assert any(s >= 3 for s in sleeps), f"Crawl-delay not honored: {sleeps[:10]}"
    assert any(s >= 5 for s in sleeps), "503 backoff missing"
    # re-run = no duplicates
    assert sc.backfill(datetime(2026, 8, 1, tzinfo=UTC), datetime(2026, 9, 30, tzinfo=UTC), fetcher=fetcher, store=sc.ArticleStore()) == 0
    print("scraper backfill OK:", n, "articles")

    # robots unreachable -> disallow all
    f2 = sc.PoliteFetcher(session=FakeSession({f"{B}/robots.txt": FakeResp(500)}), sleep=lambda s: None, min_interval=0)
    assert f2.get(f"{B}/2026/09/05/etf-approved") is None
    print("robots 5xx -> disallow OK")

    # live mode via RSS: available = first_seen; future-dated publisher clock clamped
    now = datetime(2026, 9, 20, 10, 0, tzinfo=UTC)
    rss = f"""<?xml version="1.0"?><rss version="2.0"><channel>
<item><title>Exchange hacked for $200M</title><link>{B}/2026/09/20/hack</link><pubDate>Sun, 20 Sep 2026 09:58:00 +0000</pubDate></item>
<item><title>Future dated</title><link>{B}/2026/09/20/future</link><pubDate>Sun, 20 Sep 2026 11:00:00 +0000</pubDate></item>
</channel></rss>""".encode()
    routes[f"{B}/feed"] = FakeResp(200, rss)
    routes[f"{B}/2026/09/20/hack"] = FakeResp(200, article_html("Exchange hacked for $200M", "2026-09-20T09:58:00Z"))
    routes[f"{B}/2026/09/20/future"] = FakeResp(200, article_html("Future dated", "2026-09-20T11:00:00Z"))
    new = sc.live_poll_once(fetcher=fetcher, store=store, now_fn=lambda: now)
    assert len(new) == 2
    by = {r["title"]: r for r in new}
    assert sc.parse_dt(by["Exchange hacked for $200M"]["available_ts"]) == now
    assert sc.parse_dt(by["Exchange hacked for $200M"]["published_ts"]) == datetime(2026, 9, 20, 9, 58, tzinfo=UTC)
    assert sc.parse_dt(by["Future dated"]["published_ts"]) == now
    print("live RSS OK")

    # GDELT backfill (headlines only) — seendate is the timestamp
    gd = {"articles": [{"url": "https://other.example.org/btc-news", "title": "BTC rallies on Fed cut", "seendate": "20260915T143000Z"}]}
    import urllib.parse
    params = {"query": f"({config.GDELT_QUERY}) sourcelang:english", "mode": "artlist", "format": "json", "maxrecords": "250",
              "sort": "datedesc", "startdatetime": "20260915120000", "enddatetime": "20260915180000"}
    gurl = config.GDELT_DOC_API_URL + "?" + urllib.parse.urlencode(params)
    gsess = FakeSession({gurl: FakeResp(200, json.dumps(gd).encode()),
                         "https://api.gdeltproject.org/robots.txt": FakeResp(404)})
    gf = sc.PoliteFetcher(session=gsess, sleep=lambda s: None, min_interval=0)
    ng = sc.gdelt_backfill(datetime(2026, 9, 15, 12, tzinfo=UTC), datetime(2026, 9, 15, 18, tzinfo=UTC), fetch_bodies=False,
                           fetcher=gf, store=store)
    assert ng == 1, ng
    print("gdelt backfill OK")
finally:
    config.SCRAPER_SITES = orig_sites

# ---------------------------------------------------------------- LLM parsing / scoring
assert ls.parse_llm_json('Sure! ```json\n{"relevance": 0.9, "sentiment": -1.7, "impact": 0.8, "category": "Hack/Exploit", "novelty": 1}\n```') == \
    {"relevance": 0.9, "sentiment": -1.0, "impact": 0.8, "novelty": 1.0, "category": "hack_exploit"}
assert ls.parse_llm_json("{'relevance': 0.5, 'sentiment': 0.2,}")["sentiment"] == 0.2
assert ls.parse_llm_json("no json here") is None
assert ls.parse_llm_json('{"relevance": "high", "sentiment": 0.3}') is None
assert ls.parse_llm_json('{"relevance": 0.4, "sentiment": 0.3, "category": "weird"}')["category"] == "other"
print("LLM JSON parsing OK")

seen_prompts = []


def fake_generate(batch):
    seen_prompts.extend(batch)
    out = []
    for msgs in batch:
        u = msgs[-1]["content"]
        if "Future dated" in u:
            out.append("I cannot answer")  # unparseable -> tier-1 fallback for this item
        elif "hacked" in u:
            out.append('{"relevance":0.95,"sentiment":-0.8,"impact":0.9,"category":"hack_exploit","novelty":0.9}')
        else:
            out.append('{"relevance":0.8,"sentiment":0.6,"impact":0.7,"category":"etf_institutional","novelty":0.8}')
    return out


from inference.sentiment_scorer import LexiconFallbackScorer  # noqa: E402

scorer = ls.TwoTierScorer(tier1=LexiconFallbackScorer(), tier2=ls.LlmNewsScorer(generate_fn=fake_generate, batch_size=2))
n_scored = ls.score_pending(scorer)
assert n_scored == 6, n_scored
assert ls.score_pending(scorer) == 0  # idempotent
assert all(m[0]["role"] == "system" for m in seen_prompts) and any("Article:\n" in m[1]["content"] for m in seen_prompts)
arts = nf.load_scored_articles()
assert arts.height == 6
row = {r["title"]: r for r in arts.to_dicts()}
assert row["Future dated"]["llm_ok"] is False and row["Exchange hacked for $200M"]["category"] == "hack_exploit"
assert abs(row["Exchange hacked for $200M"]["sentiment"] - (0.7 * -0.8 + 0.3 * -1.0)) < 1e-9  # blend with lexicon "hacked" = -1
print("two-tier scoring OK")

# ---------------------------------------------------------------- features: point-in-time + brute force check
rng = np.random.default_rng(0)
t0 = datetime(2026, 1, 1, tzinfo=UTC)
titles_pool = ["Bitcoin ETF sees record inflows", "SEC delays decision on spot ether fund", "Exchange suffers hot wallet exploit",
               "Fed signals rate cut in december", "Miners capitulate as hashprice falls", "Major bank launches custody service"]
syn = []
for i in range(400):
    at = t0 + timedelta(seconds=int(rng.integers(0, 4 * 86400)))
    title = titles_pool[i % 6] + f" {i // 6}" if rng.random() > 0.2 else titles_pool[i % 6]  # some near-dups
    syn.append({"available_ts": at.isoformat(), "title": title, "sentiment": float(rng.uniform(-1, 1)),
                "relevance": float(rng.uniform(0, 1)), "impact": float(rng.uniform(0, 1)), "novelty": float(rng.uniform(0, 1)),
                "category": ["regulation", "hack_exploit", "etf_institutional", "macro", "other"][i % 5], "llm_ok": bool(i % 3)})
syn_df = pl.DataFrame(syn, schema=nf._SCHEMA)

bar_ts = np.array([np.datetime64("2026-01-01T00:00:00") + np.timedelta64(60 * k, "s") for k in range(0, 5 * 1440, 7)]).astype("datetime64[us]")
bars = pl.DataFrame({"ts": bar_ts, "x": np.arange(len(bar_ts))})
out = nf.attach_news_features(bars, syn_df)
assert out.columns == ["ts", "x", *nf.NEWS_FEATURE_COLUMNS] and out.height == bars.height
assert all(out[c].dtype == pl.Float32 for c in nf.NEWS_FEATURE_COLUMNS)

# brute force for a sample of bars
arr = nf._articles_to_arrays(syn_df)
T_all = nf.bar_ts_to_ns(bars["ts"])
for k in rng.choice(len(T_all), 60, replace=False):
    T = T_all[k]
    vis = (arr["ts"] <= T) & arr["is_first"]
    for name, secs in nf.WINDOWS.items():
        m = vis & (arr["ts"] > T - secs * 10**9)
        w, s = arr["weight"][m], arr["sent"][m]
        exp_n = float((arr["rel"][m] >= 0.3).sum())
        exp_mean = float((w * s).sum() / w.sum()) if w.sum() > 1e-9 else 0.0
        covered = arr["ts"][0] <= T <= arr["ts"][-1] + nf.COVERAGE_TAIL_SECONDS * 10**9
        if not covered:
            exp_n, exp_mean = 0.0, 0.0
        assert abs(out[f"news_n_{name}"][int(k)] - exp_n) < 1e-3, (name, k)
        assert abs(out[f"news_sent_{name}"][int(k)] - exp_mean) < 1e-4, (name, k, out[f"news_sent_{name}"][int(k)], exp_mean)
    # decay brute force
    m = vis
    tau = 1800 / np.log(2)
    exp_decay = float((arr["weight"][m] * arr["sent"][m] * np.exp(-(T - arr["ts"][m]) / 1e9 / tau)).sum())
    if arr["ts"][0] <= T <= arr["ts"][-1] + nf.COVERAGE_TAIL_SECONDS * 10**9:
        assert abs(out["news_decay_30m"][int(k)] - exp_decay) < 1e-3, (k, out["news_decay_30m"][int(k)], exp_decay)
assert (~arr["is_first"]).sum() > 0, "dedup found no near-duplicates"
print("features brute-force match OK; dups:", int((~arr["is_first"]).sum()))

# strict point-in-time: one article at 12:00:30 must be invisible at 12:00:00 and visible at 12:01:00
one = pl.DataFrame([{"available_ts": "2026-01-01T12:00:30+00:00", "title": "x big news", "sentiment": 0.9, "relevance": 1.0,
                     "impact": 1.0, "novelty": 1.0, "category": "macro", "llm_ok": True}], schema=nf._SCHEMA)
b2 = pl.DataFrame({"ts": np.array(["2026-01-01T12:00:00", "2026-01-01T12:01:00", "2026-01-01T12:20:00", "2026-01-01T13:30:00"],
                                  dtype="datetime64[us]")})
o2 = nf.attach_news_features(b2, one)
assert o2["news_n_15m"].to_list() == [0.0, 1.0, 0.0, 0.0], o2["news_n_15m"].to_list()
assert o2["news_n_1h"].to_list() == [0.0, 1.0, 1.0, 0.0]
assert o2["news_coverage"].to_list() == [0.0, 1.0, 1.0, 1.0]
assert o2["news_cat_macro_4h"][1] > 0
# tz-aware bars give identical results
o3 = nf.attach_news_features(b2.with_columns(pl.col("ts").dt.replace_time_zone("UTC")), one)
assert o3["news_n_1h"].to_list() == o2["news_n_1h"].to_list()
# empty store -> identical schema, zeros
o4 = nf.attach_news_features(b2, nf._empty_articles())
assert o4.columns == o2.columns and float(o4["news_coverage"].sum()) == 0.0
print("point-in-time + empty-schema OK")

# ---------------------------------------------------------------- model integration on synthetic bars
from common.feature_contract import COLUMN_DTYPES, FEATURE_COLUMNS  # noqa: E402
import common.oof as common_oof  # noqa: E402
import models.sentiment_branch.train as sb  # noqa: E402
import models.gbm_family.train as gb  # noqa: E402
from inference import checkpoint_loaders as cl  # noqa: E402

N_DAYS, STEP = 4, 60
n_rows = N_DAYS * 86400 // STEP
ts = (np.datetime64("2026-03-01T00:00:00") + np.arange(n_rows) * np.timedelta64(STEP, "s")).astype("datetime64[us]")
cols = {"ts": ts}
for c, dt in COLUMN_DTYPES.items():
    if c == "ts":
        continue
    if dt.startswith("Boolean"):
        cols[c] = rng.random(n_rows) > 0.5
    elif dt.startswith(("Int", "UInt")):
        cols[c] = rng.integers(0, 3, n_rows)
    else:
        cols[c] = rng.normal(size=n_rows).astype(np.float32)
bars = pl.DataFrame(cols)
# news store for this window: sentiment drives the label
news_rows = []
for i in range(600):
    at = datetime(2026, 3, 1, tzinfo=UTC) + timedelta(seconds=int(rng.integers(0, N_DAYS * 86400)))
    news_rows.append({"available_ts": at.isoformat(), "title": f"story {i} about bitcoin topic{i}", "sentiment": float(rng.choice([-0.9, 0.9])),
                      "relevance": 0.9, "impact": 0.9, "novelty": 0.9, "category": "macro", "llm_ok": True})
news_df = pl.DataFrame(news_rows, schema=nf._SCHEMA)
scored_dir = config.NEWS_ARTICLES_ROOT / "scored"
shutil.rmtree(scored_dir)
scored_dir.mkdir(parents=True)
(scored_dir / "scored_2026-03-01.jsonl").write_text("\n".join(json.dumps(r) for r in news_rows) + "\n", encoding="utf-8")
feat = nf.compute_news_features(nf.bar_ts_to_ns(bars["ts"]), news_df)
p = feat["news_decay_30m"]
label = np.where(p > 0.3, 1, np.where(p < -0.3, -1, 0))
flip = rng.random(n_rows) < 0.2
label = np.where(flip, rng.integers(-1, 2, n_rows), label)
bars = bars.with_columns(pl.Series("label_triple_barrier", label).cast(bars["label_triple_barrier"].dtype),
                         pl.lit(True).alias("label_is_valid"))

saved = {"sb_load": sb.load_days, "sb_split": sb.purged_walk_forward_splits, "gb_load": gb.load_days,
         "gb_split": gb.purged_walk_forward_splits, "save_oof": common_oof.save_oof, "bpd": config.BARS_PER_DAY,
         "sb_ck": sb.CHECKPOINT_DIR, "sb_ab": sb.ABLATION_CHECKPOINT_DIR, "gb_ck": gb.CHECKPOINT_DIR}
oofs = {}
try:
    orig_split = sb.purged_walk_forward_splits
    split_wrap = lambda df, n_folds, val_size: orig_split(df, n_folds=n_folds, val_size=val_size, purge_bars=5, embargo_bars=5)  # noqa: E731
    sb.load_days = gb.load_days = lambda max_days=None, **k: bars
    sb.purged_walk_forward_splits = gb.purged_walk_forward_splits = split_wrap
    common_oof.save_oof = lambda name, ts, proba: oofs.__setitem__(name, proba)
    config.BARS_PER_DAY = 86400 // STEP
    sb.CHECKPOINT_DIR = TMP / "sb_ck"
    sb.ABLATION_CHECKPOINT_DIR = TMP / "sb_ck" / "ablation"
    gb.CHECKPOINT_DIR = TMP / "gb_ck"

    m = sb.run(val_days=1, max_days=None, upload=False, with_news=True)
    assert m["feature_columns"][-len(nf.NEWS_FEATURE_COLUMNS):] == nf.NEWS_FEATURE_COLUMNS and not m["is_placeholder_proxy"]
    print(f"sentiment_branch --with-news accuracy={m['accuracy']:.3f}")
    abl = sb.run_ablation(val_days=1, max_days=None, with_news=True)
    print(f"ablation with_news={abl['with']['accuracy']:.3f} without={abl['without']['accuracy']:.3f}")
    assert abl["with"]["accuracy"] > abl["without"]["accuracy"] + 0.1, abl
    assert abl["with"]["n_features"] == len(FEATURE_COLUMNS) + len(nf.NEWS_FEATURE_COLUMNS)
    m0 = sb.run(val_days=1, max_days=None, upload=False, with_news=False)  # default path unchanged
    assert m0["is_placeholder_proxy"] and m0["feature_columns"] == sb.SENTIMENT_PROXY_FEATURES
    sb.run(val_days=1, max_days=None, upload=False, with_news=True)  # leave news checkpoint for loader test

    # live loader: reads feature list from metrics.json, attaches news on the fly
    import lightgbm as lgb
    cols_trained = cl._trained_feature_columns(sb.CHECKPOINT_DIR, cl.SENTIMENT_PROXY_FEATURES)
    pred = cl._SentimentBranchPredictor(lgb.Booster(model_file=str(sb.CHECKPOINT_DIR / "sentiment_proxy_classifier.txt")), cols_trained)
    recent = bars.drop([c for c in bars.columns if c.startswith("label")]).slice(3000, 50)
    proba = pred.predict(recent)
    assert proba.shape == (3,) and abs(proba.sum() - 1) < 1e-6
    assert cl._trained_feature_columns(TMP / "nope", ["a"]) == ["a"]
    print("live sentiment predictor with news OK:", np.round(proba, 3))

    r = gb.run(n_folds=2, val_days=1, max_days=None, upload=False, with_news=True)
    assert r["with_news"] and r["feature_columns"] == FEATURE_COLUMNS + nf.NEWS_FEATURE_COLUMNS
    gpred = cl._GbmFamilyPredictor(lgb.Booster(model_file=str(gb.CHECKPOINT_DIR / "direction_classifier.txt")),
                                   cl._trained_feature_columns(gb.CHECKPOINT_DIR, FEATURE_COLUMNS))
    assert gpred.predict(recent).shape == (3,)
    print(f"gbm --with-news OK final direction acc={r['final_fold']['direction']['accuracy']:.3f}")

    # coverage guard
    shutil.rmtree(scored_dir)
    try:
        sb.load_with_news(None)
        raise AssertionError("expected ValueError for uncovered window")
    except ValueError:
        pass
    print("coverage guard OK")
finally:
    sb.load_days, sb.purged_walk_forward_splits = saved["sb_load"], saved["sb_split"]
    gb.load_days, gb.purged_walk_forward_splits = saved["gb_load"], saved["gb_split"]
    common_oof.save_oof, config.BARS_PER_DAY = saved["save_oof"], saved["bpd"]
    sb.CHECKPOINT_DIR, sb.ABLATION_CHECKPOINT_DIR, gb.CHECKPOINT_DIR = saved["sb_ck"], saved["sb_ab"], saved["gb_ck"]
    shutil.rmtree(TMP, ignore_errors=True)

print("ALL TESTS PASSED")
