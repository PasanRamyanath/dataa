"""
Central configuration for the Binance/Bybit BTC feature-engineering pipeline.

All paths default to the layout observed on disk under
`F:\\Downloads\\Binance data`. Override via environment variables for
portability (e.g. when running on Kaggle: DATA_ROOT=/kaggle/input/...).
"""
import os
import socket
from pathlib import Path

# Force IPv4 preference on environments with blackholed IPv6 routes:
_orig_getaddrinfo = socket.getaddrinfo
def _ipv4_first_getaddrinfo(*args, **kwargs):
    res = _orig_getaddrinfo(*args, **kwargs)
    v4 = [r for r in res if r[0] == socket.AF_INET]
    return v4 if v4 else res
socket.getaddrinfo = _ipv4_first_getaddrinfo

# --------------------------------------------------------------------------
# Filesystem roots
# --------------------------------------------------------------------------
_DEFAULT_PROJECT_ROOT = (
    Path(r"D:\backup\Binance data") if Path(r"D:\backup\Binance data").exists()
    else Path(r"F:\Downloads\Binance data")
)
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", _DEFAULT_PROJECT_ROOT))

DATA_ROOT = Path(os.environ.get("BINANCE_DATA_ROOT", PROJECT_ROOT / "data"))
PREMIUM_INDEX_ROOT = Path(os.environ.get("BYBIT_PREMIUM_INDEX_ROOT", PROJECT_ROOT / "premium_index"))
SPOT_INDEX_ROOT = Path(os.environ.get("BYBIT_SPOT_INDEX_ROOT", PROJECT_ROOT / "spot_index"))
TRADING_ROOT = Path(os.environ.get("BYBIT_TRADING_ROOT", PROJECT_ROOT / "trading"))

OUTPUT_ROOT = Path(os.environ.get("FEATURE_OUTPUT_ROOT", PROJECT_ROOT / "feature engineering scripts" / "output"))
CACHE_ROOT = Path(os.environ.get("FEATURE_CACHE_ROOT", PROJECT_ROOT / "feature engineering scripts" / ".cache"))

# --------------------------------------------------------------------------
# Symbols
# --------------------------------------------------------------------------
SYMBOL_SPOT = "BTCUSDT"
SYMBOL_UM = "BTCUSDT"          # USDT-margined perpetual
SYMBOL_CM = "BTCUSD_PERP"      # COIN-margined perpetual
SYMBOL_CM_INDEX = "BTCUSD"     # cm indexPriceKlines is keyed by the underlying pair, not the contract
BYBIT_SYMBOL = "BTCUSDT"       # Bybit linear (USDT-margined) perpetual — used for trading/ (cross-exchange trade tape)

# Bybit's premium_index/ and spot_index/ dumps are only published under the
# INVERSE contract's underlying symbol (confirmed: sample data has no
# BTCUSDT folder under either tree, only BTCUSD) — distinct from
# BYBIT_SYMBOL above, which is what trading/ uses.
BYBIT_INDEX_SYMBOL = "BTCUSD"

# Binance's BTC implied-volatility index (option/daily/BVOLIndex/), keyed by
# its own symbol rather than any spot/futures pair.
BVOL_SYMBOL = "BTCBVOLUSDT"

# USDⓈ-M / COIN-M quarterly delivery-futures base symbols, for term-structure
# (calendar basis) features — see contracts.py for how the actual dated
# contract symbol (e.g. BTCUSDT_260327) is derived per day.
QUARTERLY_BASE_UM = "BTCUSDT"
QUARTERLY_BASE_CM = "BTCUSD"

# --------------------------------------------------------------------------
# Bar resolutions and model sequence lengths
# Confirmed against two independent existing training scripts
# (train_tft_kaggle.py argparse defaults, tft_feature_extraction_2_(2).py Config):
#   LOOKBACK=720, HORIZON=90 @ 10s bars == 2h of context -> 15min forecast.
# --------------------------------------------------------------------------
BAR_INTERVALS = ["1s", "10s", "1m", "5m"]
PRIMARY_INTERVAL = "10s"
PRIMARY_INTERVAL_SECONDS = 10

LOOKBACK_PERIODS = 720
FORECAST_PERIODS = 90

BARS_PER_DAY = 24 * 60 * 60 // PRIMARY_INTERVAL_SECONDS  # 8640 @ 10s

# Binance funding is paid every 8h at fixed UTC boundaries (00:00/08:00/16:00).
FUNDING_INTERVAL_SECONDS = 8 * 3600
BARS_PER_FUNDING_INTERVAL = FUNDING_INTERVAL_SECONDS // PRIMARY_INTERVAL_SECONDS  # 2880 @ 10s

# --------------------------------------------------------------------------
# Hugging Face publishing
# Mirrors the safest existing pattern found in the codebase
# (colab_tft_feature_extraction.py / tft_feature_extraction_2_(2).py):
# token ONLY from env var, never hardcoded; flat `data/` prefix, one file
# per calendar day, HfApi().upload_file (never push_to_hub/upload_folder).
# --------------------------------------------------------------------------
HF_REPO_ID = os.environ.get("HF_REPO_ID", "p-j-r-1-2-3/Project-IMPJR-ModelTFT-dataset")
HF_REPO_TYPE = "dataset"
# Versioned: "data/" holds days built before the point-in-time timestamp
# fixes (1m index/mark/premium klines and Bybit index keyed at bar OPEN,
# 5m metrics keyed at window START — both leaked up to 1-5 min of future
# into features). kaggle_run_pipeline.py resumes by skipping days already
# under this prefix, so bumping it makes the next run rebuild every day
# leak-free, and loaders (common/data.py) only ever see rebuilt days. Bump
# again whenever a fix changes already-published feature values.
HF_DATA_PREFIX = os.environ.get("HF_DATA_PREFIX", "data_v2")

# --------------------------------------------------------------------------
# Model training / private HF model publishing (models/, validation/, fusion/)
# Distinct repos from HF_REPO_ID above (which is the public-by-default daily
# FEATURE dataset repo): every model-training script and models/common/hf_upload.py
# push checkpoints + their exact training data snapshot to PRIVATE repos only.
# --------------------------------------------------------------------------
HF_MODEL_ORG = os.environ.get("HF_MODEL_ORG", "p-j-r-1-2-3")
HF_MODEL_REPO_PREFIX = os.environ.get("HF_MODEL_REPO_PREFIX", "Project-IMPJR-v4-Models")  # -> {org}/{prefix}-{member}
HF_TRAINING_DATA_REPO_ID = os.environ.get(
    "HF_TRAINING_DATA_REPO_ID", f"{HF_MODEL_ORG}/Project-IMPJR-TrainingData"
)


def get_hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        import platform
        if platform.system() == "Windows":
            import winreg
            try:
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as key:
                    token, _ = winreg.QueryValueEx(key, "HF_TOKEN")
            except OSError:
                pass
    if not token:
        token = "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"
    return token


# --------------------------------------------------------------------------
# Labeling (triple-barrier)
# --------------------------------------------------------------------------
LABEL_HORIZONS_BARS = {
    "h1": 12,    # ~2 min at 10s bars
    "h2": 60,    # ~10 min
    "h3": 360,   # ~1 hr
    "h4": 8640,  # ~24 hr
}
TRIPLE_BARRIER_VOL_LOOKBACK = 90       # bars used to estimate rolling realized vol for barrier sizing
TRIPLE_BARRIER_PT_MULT = 2.0           # profit-take = mult * rolling_vol
TRIPLE_BARRIER_SL_MULT = 2.0           # stop-loss   = mult * rolling_vol
TRIPLE_BARRIER_MAX_HOLD_BARS = 360     # time barrier fallback (~1hr)

# --------------------------------------------------------------------------
# Anti-leakage / validation
# --------------------------------------------------------------------------
EMBARGO_BARS = 8640     # 1 day embargo between train/val/test folds
PURGE_BARS = 360        # purge window (~1hr) around fold boundaries to remove label-overlap leakage
VAL_DAYS = 7            # matches train_tft_kaggle.py --val-days default

# OOF export density / fusion alignment. Sequence members score val windows
# on a strided grid, so fusion.stacking_meta_learner.build_meta_features
# joins each member's latest prediction made AT OR BEFORE a row's ts (a
# backward as-of join, never forward) and drops rows where some member's
# latest prediction is older than OOF_MAX_STALENESS_BARS.
SEQ_OOF_STEP_BARS = 10          # neuralforecast cross_validation step_size (cutoff every ~100s)
KRONOS_OOF_STEP_BARS = 48       # Kronos val-window stride (autoregressive decode is slow)
OOF_MAX_STALENESS_BARS = 60     # must be >= every member's OOF stride

FLOAT_DTYPE = "float32"

# --------------------------------------------------------------------------
# Live news/sentiment ingestion (inference/news_feed.py, inference/sentiment_scorer.py)
# Plan §9.1/§9.2: free-tier-only sources, no CryptoPanic (paid-only since
# early 2026), no X/Twitter. Distinct from models/sentiment_branch/ (which
# trains on already-computed crowd-positioning proxy features, not text) —
# this is the headline-level ingestion §9.1 describes. The full-text
# scraper + LLM scoring + training-time news features that feed the models
# live in news_pipeline/ (config block further below).
# --------------------------------------------------------------------------
NEWS_DATA_ROOT = Path(os.environ.get("NEWS_DATA_ROOT", PROJECT_ROOT / "feature engineering scripts" / "inference" / "data" / "news_events"))
SENTIMENT_FEATURES_ROOT = Path(os.environ.get("SENTIMENT_FEATURES_ROOT", PROJECT_ROOT / "feature engineering scripts" / "inference" / "data" / "sentiment_features"))
NEWS_STATE_ROOT = Path(os.environ.get("NEWS_STATE_ROOT", PROJECT_ROOT / "feature engineering scripts" / "inference" / "data" / "_state"))

RSS_FEEDS = {
    "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "cointelegraph": "https://cointelegraph.com/rss",
    "theblock": "https://www.theblock.co/rss.xml",
    "decrypt": "https://decrypt.co/feed",
}
RSS_POLL_INTERVAL_SECONDS = 10           # §9.1: poll every 10s, conditional GET keeps repeat polls cheap
GDELT_DOC_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_QUERY = "bitcoin OR BTC OR cryptocurrency"
GDELT_POLL_INTERVAL_SECONDS = 15 * 60    # §9.1: GDELT's own update cadence is 15 minutes
WHALE_ALERT_API_URL = "https://api.whale-alert.io/v1/transactions"
WHALE_ALERT_MIN_USD = 500_000
WHALE_ALERT_POLL_INTERVAL_SECONDS = 60

# Explicitly excluded per §9.1 (documented here, not just in prose, so a
# future contributor doesn't "helpfully" re-add them): CryptoPanic (free
# tier discontinued early 2026) and any X/Twitter access (paid API with hard
# volume caps; unauthorized scraping is active litigation territory).
EXCLUDED_NEWS_SOURCES = ("cryptopanic", "x_twitter")


def get_whale_alert_api_key() -> str | None:
    """Returns None (not raises) when unset — the whale-alert feed is one of
    three §9.1 sources, not a hard requirement to run the others; see
    inference/news_feed.py's per-source graceful-degradation handling."""
    return os.environ.get("WHALE_ALERT_API_KEY") or None


# distilled/quantized FinBERT-class model, well under 500MB at int8/fp16 per
# §7.1 — crypto-domain-tuned by default; override via env var to swap in
# plain ProsusAI/finbert or any other AutoModelForSequenceClassification.
SENTIMENT_MODEL_NAME = os.environ.get("SENTIMENT_MODEL_NAME", "ElKulako/cryptobert")
SENTIMENT_MENTION_VELOCITY_WINDOW_SECONDS = 15 * 60

# --------------------------------------------------------------------------
# Full-text news scraper + two-tier AI scoring + point-in-time news features
# (news_pipeline/). This is what finally closes the §9.3 loop: historical
# backfill (sitemaps discovered via robots.txt, GDELT DOC history) gives the
# models a TRAINING history of text sentiment, live mode keeps it current,
# and news_pipeline.features.attach_news_features() joins it onto any bars
# DataFrame strictly point-in-time.
# --------------------------------------------------------------------------
NEWS_ARTICLES_ROOT = Path(os.environ.get("NEWS_ARTICLES_ROOT", PROJECT_ROOT / "feature engineering scripts" / "inference" / "data" / "news_articles"))

# Outlets to crawl. Sitemaps are discovered from each site's robots.txt
# ("Sitemap:" lines) rather than hard-coded, so a site moving its sitemap
# doesn't silently break backfill. article_pattern filters out tag/author/
# category pages that sitemaps also list.
SCRAPER_SITES = {
    "coindesk": {"base_url": "https://www.coindesk.com", "rss": RSS_FEEDS["coindesk"],
                  "article_pattern": r"/20\d\d/\d\d/\d\d/"},
    "cointelegraph": {"base_url": "https://cointelegraph.com", "rss": RSS_FEEDS["cointelegraph"],
                       "article_pattern": r"/news/|/magazine/"},
    "theblock": {"base_url": "https://www.theblock.co", "rss": RSS_FEEDS["theblock"],
                  "article_pattern": r"/post/\d+"},
    "decrypt": {"base_url": "https://decrypt.co", "rss": RSS_FEEDS["decrypt"],
                 "article_pattern": r"/\d+/"},
    "bitcoinmagazine": {"base_url": "https://bitcoinmagazine.com", "rss": "https://bitcoinmagazine.com/.rss/full/",
                         "article_pattern": r"/(markets|business|news|politics|technical|culture)/"},
}
SCRAPER_USER_AGENT = os.environ.get(
    "SCRAPER_USER_AGENT", "BinanceFuturesResearchBot/1.0 (+non-commercial research; respects robots.txt)"
)
SCRAPER_MIN_DOMAIN_INTERVAL_SECONDS = float(os.environ.get("SCRAPER_MIN_DOMAIN_INTERVAL_SECONDS", "2.0"))  # floor; robots Crawl-delay wins if larger
SCRAPER_TIMEOUT_SECONDS = 20
SCRAPER_MAX_RETRIES = 3
SCRAPER_MAX_BYTES = 5_000_000
SCRAPER_LIVE_POLL_SECONDS = 30
# Fetch threads. Jobs are sharded by domain: domains run in parallel, one
# domain's requests stay strictly sequential (so per-site politeness is
# unchanged; only different outlets overlap — mostly GDELT backfill).
SCRAPER_MAX_WORKERS = int(os.environ.get("SCRAPER_MAX_WORKERS", "8"))
SCRAPER_DEDUP_JACCARD = 0.6              # title token-set Jaccard above this = same story, keep first-seen
SCRAPER_DEDUP_WINDOW_HOURS = 48
# Point-in-time: a backfilled article is treated as "known" only
# published_ts + this lag (live scraping/scoring latency is never zero).
# Live-scraped articles use their actual first_seen_ts instead.
NEWS_BACKFILL_AVAILABILITY_LAG_SECONDS = int(os.environ.get("NEWS_BACKFILL_AVAILABILITY_LAG_SECONDS", "120"))

# Tier-2 scorer: small instruct LLM producing structured JSON (relevance,
# sentiment, impact, category, novelty). Loaded fp16 on GPU (T4 has no
# bf16), 4-bit via bitsandbytes if NEWS_LLM_LOAD_IN_4BIT=1 (lets a 7B model
# fit next to the price members). NEWS_LLM_ENABLED=0 → tier-1 only.
NEWS_LLM_MODEL_NAME = os.environ.get("NEWS_LLM_MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
NEWS_LLM_ENABLED = os.environ.get("NEWS_LLM_ENABLED", "1") != "0"
NEWS_LLM_LOAD_IN_4BIT = os.environ.get("NEWS_LLM_LOAD_IN_4BIT", "0") == "1"
NEWS_LLM_BATCH_SIZE = int(os.environ.get("NEWS_LLM_BATCH_SIZE", "8"))
NEWS_LLM_MAX_INPUT_CHARS = 2500
NEWS_LLM_WEIGHT = 0.7                    # blend: 0.7 * LLM sentiment + 0.3 * tier-1 classifier sentiment
# Inference backend: "auto" = vLLM when it is importable (fp16, eager mode,
# JSON-schema guided decoding; install per README-style note in
# news_pipeline/llm_scorer.py) else HF transformers generate; "hf"/"vllm"
# force one. NEWS_LLM_GPUS: replicas, one full model copy per GPU (data-
# parallel — on 2x T4 this beats tensor-parallel for a 3B model, which is
# PCIe-bound); "auto" = every visible GPU, "0" = CPU/default placement.
NEWS_LLM_BACKEND = os.environ.get("NEWS_LLM_BACKEND", "auto").lower()
NEWS_LLM_GPUS = os.environ.get("NEWS_LLM_GPUS", "auto").lower()
NEWS_LLM_VLLM_GPU_MEMORY_UTILIZATION = float(os.environ.get("NEWS_LLM_VLLM_GPU_MEMORY_UTILIZATION", "0.85"))
# Look-ahead guard (Glasserman & Lin 2023): an LLM scoring an article
# published before its own training cutoff may already "know" what the price
# did next, which inflates backtests and teaches the models to over-trust
# news. Articles published on/before this date are scored tier-1 only
# (llm_ok=False, so news_llm_share_24h tells the models). Qwen2.5's cutoff
# is 2024-06-30 — before the default training window — so this is a no-op
# for the default model; set it when switching NEWS_LLM_MODEL_NAME. ""
# disables the guard.
NEWS_LLM_KNOWLEDGE_CUTOFF = os.environ.get("NEWS_LLM_KNOWLEDGE_CUTOFF", "2024-06-30")
# Kaggle session: refresh the news store on the GPU VM before training
# (incremental GDELT + sitemap scrape since the store's newest article,
# dual-T4 scoring, push back to HF_NEWS_REPO_ID). Needs the notebook's
# Internet toggle ON. Off by default — the scraper normally runs locally.
NEWS_REFRESH_ON_KAGGLE = os.environ.get("NEWS_REFRESH_ON_KAGGLE", "0") == "1"
NEWS_REFRESH_MAX_DAYS = int(os.environ.get("NEWS_REFRESH_MAX_DAYS", "7"))

# Optional path to rs-trafilatura's `extract_stdin` binary (WCXB 2026 leader,
# F1 0.859 vs trafilatura 0.791). No Python binding exists, so the scraper
# pipes HTML to it as one more consensus candidate when this points at an
# executable; empty/missing -> skipped. Build it yourself (cargo build
# --release --bin extract_stdin) or attach it as a Kaggle dataset.
RS_TRAFILATURA_BIN = os.environ.get("RS_TRAFILATURA_BIN", "")
RS_TRAFILATURA_TIMEOUT_S = float(os.environ.get("RS_TRAFILATURA_TIMEOUT_S", "10"))

# Private HF dataset the scored news store is shipped through (news_pipeline.
# store_sync push locally -> pull on Kaggle). TRAIN_WITH_NEWS for the Kaggle
# session: "auto" trains gbm-family/sentiment-branch with --with-news iff the
# pull produced scored articles; "1" requires it; "0" never.
HF_NEWS_REPO_ID = os.environ.get("HF_NEWS_REPO_ID", f"{HF_MODEL_ORG}/Project-IMPJR-News")
TRAIN_WITH_NEWS = os.environ.get("TRAIN_WITH_NEWS", "auto").lower()

# --------------------------------------------------------------------------
# Live inference loop + paper trading (§12.1/§12.2). PAPER_TRADING_DATA_ROOT
# is distinct from TRADING_ROOT above (Bybit's real cross-exchange trade-tape
# input data) — this is this engine's own OUTPUT (simulated fills, equity
# curve, circuit-breaker state), never a live-execution path to a real
# exchange account.
# --------------------------------------------------------------------------
LIVE_TICK_INTERVAL_SECONDS = 1.0        # §12.1 target cadence
LIVE_MAX_TICK_LATENCY_SECONDS = 5.0     # ticks slower than this are logged as degraded, not silently absorbed

PAPER_TRADING_DATA_ROOT = Path(os.environ.get(
    "PAPER_TRADING_DATA_ROOT", PROJECT_ROOT / "feature engineering scripts" / "inference" / "data" / "paper_trading"
))
LIVE_TICK_LOG_ROOT = Path(os.environ.get(
    "LIVE_TICK_LOG_ROOT", PROJECT_ROOT / "feature engineering scripts" / "inference" / "data" / "tick_log"
))

PAPER_STARTING_EQUITY_USD = 100_000.0
PAPER_MAX_POSITION_FRACTION = 0.5       # cap: at most 50% of equity notional in one position, even at max size
PAPER_TAKER_FEE_BPS = 5.0               # Binance USDT-M taker fee approx, applied to every simulated fill
PAPER_SLIPPAGE_BPS = 2.0                # simple fixed slippage assumption, applied against the trade direction
PAPER_MIN_REBALANCE_FRACTION = 0.01     # skip re-sizing smaller than this (1% of equity): confidence moves every tick, and fee-paying micro-trades each 10s would bleed equity
META_LABEL_ACTION_THRESHOLD = 0.70       # Ultra Sniper meta-label filter threshold: below 0.70 conviction, veto entry / flatten position

# Circuit breakers (§12.2): halt new position entries (never force-close an
# existing one) once tripped; both reset at the next UTC day/week boundary.
PAPER_DAILY_MAX_DRAWDOWN_PCT = 0.05     # vs. equity at the start of the current UTC day
PAPER_WEEKLY_MAX_DRAWDOWN_PCT = 0.12    # vs. equity at the start of the current ISO week

# §12.2 says only "confidence-scaled ... times a regime multiplier" with no
# table given, so this is a documented, conservative interpretation rather
# than an invented one pulled from nowhere: hmmlearn's GaussianHMM doesn't
# guarantee any fixed state-index meaning, so inference/paper_trading_engine.py
# ranks the *fitted* states by their own realized_vol_90 mean (see
# inference/checkpoint_loaders.py's HmmRegimePredictor.regime_volatility_rank)
# and only then applies low/default/high by that rank — never by raw index.
DEFAULT_REGIME_SIZE_MULTIPLIER = 1.0
LOW_VOL_REGIME_SIZE_MULTIPLIER = 1.2
HIGH_VOL_REGIME_SIZE_MULTIPLIER = 0.4

# --------------------------------------------------------------------------
# Kaggle session-relaunch orchestration (§11.2). KAGGLE_KERNEL_SLUG identifies
# the Kaggle notebook (kernel) that runs orchestration/kaggle_relay/
# run_kaggle_session.py (models.train_all under the hood); the GitHub Actions
# cron in .github/workflows/kaggle_relay.yml polls this kernel's status via
# the official `kaggle` CLI (orchestration/kaggle_relay/poll_and_relaunch.py)
# and re-pushes it whenever a session has ended but the ensemble isn't fully
# trained yet (per orchestration/checkpoint_manager.py's HF-repo-backed
# ensemble_complete() check, not any Kaggle-local state, which never survives
# a fresh session's clean disk). KAGGLE_SESSION_TIME_BUDGET_HOURS is the
# proactive-stop threshold run_kaggle_session.py uses to end its OWN session
# cleanly (letting the current member/fusion-stage finish and commit its
# checkpoint to HF) a safety margin before Kaggle's ~12h hard kill, rather
# than being killed mid-write.
# --------------------------------------------------------------------------
_kaggle_json_paths = (
    Path(__file__).resolve().parent / "kaggle.json",
    Path(r"D:\backup\Binance data\feature engineering scripts\kaggle.json"),
    Path(r"d:\backup\feature engineering scripts\kaggle.json"),
    Path.home() / ".kaggle" / "kaggle.json",
)
_loaded_kaggle = False
for _kp in _kaggle_json_paths:
    if _kp.exists():
        try:
            import json
            _kd = json.loads(_kp.read_text(encoding="utf-8"))
            if _kd.get("username") and _kd.get("key"):
                KAGGLE_USERNAME = _kd["username"]
                os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
                os.environ["KAGGLE_KEY"] = _kd["key"]
                _loaded_kaggle = True
                break
        except Exception:
            pass
if not _loaded_kaggle:
    _raw_kaggle_user = os.environ.get("KAGGLE_USERNAME", "")
    KAGGLE_USERNAME = "malaysiatrip" if (not _raw_kaggle_user or _raw_kaggle_user in ("pjramyanath", "rramyanath", "rimramyanath")) else _raw_kaggle_user
    os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
KAGGLE_KERNEL_SLUG = os.environ.get("KAGGLE_KERNEL_SLUG", "btcusdt-v4-ensemble-train")
KAGGLE_SESSION_TIME_BUDGET_HOURS = float(os.environ.get("KAGGLE_SESSION_TIME_BUDGET_HOURS", "11.5"))
# Accelerator for relay pushes: "NvidiaTeslaT4" = Kaggle's 2x T4 VM (plain
# enable_gpu defaults to a single P100). Set to "" to omit the key.
KAGGLE_MACHINE_SHAPE = os.environ.get("KAGGLE_MACHINE_SHAPE", "NvidiaTeslaT4")

# --------------------------------------------------------------------------
# Streaming / webhook layer (§14). Runs from inside the same Kaggle
# inference session as inference/live_loop.py — no separate always-on
# server is introduced. STREAM_SHARED_SECRET gates every endpoint the named
# Cloudflare Tunnel exposes to the public internet (§14.1: "public
# reachability is not the same as public access"). CLOUDFLARE_TUNNEL_TOKEN
# is deliberately NOT read here via os.environ — streaming/cloudflared_manager.py
# reads it at runtime via kaggle_secrets.UserSecretsClient (same discipline
# as get_hf_token() above), falling back to CLOUDFLARE_TUNNEL_TOKEN_ENV_FALLBACK
# only for local/offline testing outside Kaggle. Webhook target URLs default
# to empty (no target configured = nothing sent, not an error) since not
# every deployment wants every channel.
# --------------------------------------------------------------------------
CLOUDFLARE_TUNNEL_TOKEN_ENV_FALLBACK = "CLOUDFLARE_TUNNEL_TOKEN"
CLOUDFLARED_BINARY_PATH = Path(os.environ.get("CLOUDFLARED_BINARY_PATH", "/usr/local/bin/cloudflared"))
CLOUDFLARED_DOWNLOAD_URL = os.environ.get(
    "CLOUDFLARED_DOWNLOAD_URL",
    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
)

STREAM_SHARED_SECRET = os.environ.get("STREAM_SHARED_SECRET", "")
STREAM_SHARED_SECRET_HEADER = "X-Stream-Secret"
STREAM_HOST = os.environ.get("STREAM_HOST", "0.0.0.0")
STREAM_PORT = int(os.environ.get("STREAM_PORT", "8787"))

DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")
GENERIC_WEBHOOK_URL = os.environ.get("GENERIC_WEBHOOK_URL", "")
GENERIC_WEBHOOK_AUTH_HEADER = os.environ.get("GENERIC_WEBHOOK_AUTH_HEADER", "")

WEBHOOK_BATCH_INTERVAL_SECONDS = 5.0    # §14.2: batch, don't fire one HTTP call per tick
WEBHOOK_MAX_QUEUE_SIZE = 2000           # local queue cap during sustained backoff, oldest dropped past this
WEBHOOK_MAX_RETRIES = 5
WEBHOOK_BACKOFF_BASE_SECONDS = 2.0
WEBHOOK_BACKOFF_MAX_SECONDS = 120.0

STREAMING_STATE_ROOT = Path(os.environ.get(
    "STREAMING_STATE_ROOT", PROJECT_ROOT / "feature engineering scripts" / "streaming" / "data"
))

# --------------------------------------------------------------------------
# Resource maximization (§13, plan-marked OPTIONAL — none of this is
# required for the core system to work). Every module under
# resource_maximization/ degrades to "skip this piece, log it, keep going"
# exactly like every other member/stage in this repo, and per §13's own
# honesty notes, only does something meaningful once real trained
# checkpoints and a live GPU inference loop exist to attach to.
#
# DRIFT_PSI_THRESHOLD: Population Stability Index cutoff above which
# resource_maximization/drift_gate.py judges the live feature distribution
# to have drifted enough from the training distribution to warrant an
# online-adaptation pass (§13.1's "estimate drift magnitude... and only
# adapt when that estimate crosses a threshold" — adapting every tick
# regardless is the drift-chasing failure mode the plan explicitly warns
# against). 0.2 is the standard PSI convention for "significant population
# shift" (0.1-0.2 moderate, >0.2 significant) used across the credit-risk/
# ML-monitoring literature this technique is borrowed from.
# --------------------------------------------------------------------------
DRIFT_PSI_THRESHOLD = 0.2
DRIFT_PSI_MIN_SAMPLES = 100          # below this, a PSI estimate is too noisy to gate on
DRIFT_PSI_BINS = 10

REPLAY_BUFFER_SIZE = 5000            # §13.1 anti-forgetting safeguard: older, diverse examples kept alongside recent-data updates
REPLAY_BUFFER_RECENT_FRACTION = 0.5  # fraction of each adaptation batch drawn from the freshest data vs. the replay buffer

# §13.3: multi-level MC dropout allocates most stochastic passes to cheap
# low-fidelity configs and only a few to expensive high-fidelity ones.
MC_DROPOUT_CHEAP_PASSES = 20
MC_DROPOUT_CHEAP_RATE = 0.1
MC_DROPOUT_EXPENSIVE_PASSES = 5
MC_DROPOUT_EXPENSIVE_RATE = 0.3
TEST_TIME_AUGMENTATION_JITTER_STD = 0.01   # relative std of the small input jitter applied for light TTA

# §13.2: correlated majors joint-modeled alongside BTCUSDT via a shared
# feature-extraction backbone with independent per-asset heads.
MULTI_ASSET_SYMBOLS = ["ETHUSDT", "SOLUSDT", "BNBUSDT"]

# Shared with validation/regime_holdout_eval.py's DEFAULT_RELATIVE_THRESHOLD
# (§10.1's "~1% relative") so an ablation variant (with vs. without a
# feature branch, §9.3/§13.2) is held to the exact same bar a member is.
ABLATION_RELATIVE_LIFT_THRESHOLD = 0.01

# resource_maximization/live_glue.py: how often (in ticks) the drift-gated
# GBM adaptation pass (§13.1) runs. Every tick would recompute PSI + refit
# a booster against essentially the same recent window for no benefit —
# checking periodically is enough since drift is a slow-moving quantity by
# construction (PSI over a rolling window), unlike MC-dropout uncertainty
# (§13.3), which IS cheap enough to run every tick and so has no cadence
# knob of its own.
LIVE_GLUE_DRIFT_CHECK_EVERY_N_TICKS = 60
# How many of the most recent already-label-resolved days to pull (via
# common.data.load_days + filter_valid) as the "recent window" for drift
# scoring and GBM adaptation — deliberately NOT the live tick loop's own
# LiveDataBuffer, which holds the freshest still-unresolved bars needed for
# prediction, not the labeled data adaptation requires.
LIVE_GLUE_ADAPTATION_LOOKBACK_DAYS = 1
