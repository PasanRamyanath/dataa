"""Offline test of inference.sentiment_scorer — forces the lexicon fallback
(no transformers/torch import, no model download, no network) and writes
synthetic news JSONL events directly rather than polling real feeds."""
import os
import sys
import json
import shutil
from datetime import datetime, timezone

SCRATCH = os.path.dirname(os.path.abspath(__file__))
NEWS_DATA = os.path.join(SCRATCH, "sentiment_smoke", "events")
NEWS_STATE = os.path.join(SCRATCH, "sentiment_smoke", "state")
SENTIMENT_FEATURES = os.path.join(SCRATCH, "sentiment_smoke", "sentiment_features")
shutil.rmtree(os.path.join(SCRATCH, "sentiment_smoke"), ignore_errors=True)
os.makedirs(NEWS_DATA, exist_ok=True)

os.environ["NEWS_DATA_ROOT"] = NEWS_DATA
os.environ["NEWS_STATE_ROOT"] = NEWS_STATE
os.environ["SENTIMENT_FEATURES_ROOT"] = SENTIMENT_FEATURES

from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from inference import sentiment_scorer as ss

# Write synthetic rss + gdelt events directly (bypassing news_feed.py's
# polling — this test is only about the scorer, not ingestion).
now = datetime.now(timezone.utc)
rss_path = os.path.join(NEWS_DATA, f"rss_coindesk_{now.strftime('%Y-%m-%d')}.jsonl")
gdelt_path = os.path.join(NEWS_DATA, f"gdelt_{now.strftime('%Y-%m-%d')}.jsonl")
whale_path = os.path.join(NEWS_DATA, f"whale_alert_{now.strftime('%Y-%m-%d')}.jsonl")

events_rss = [
    {"ts": now.isoformat(), "title": "Bitcoin surges to record high on ETF approval", "summary": "bullish rally continues"},
    {"ts": now.isoformat(), "title": "Exchange hacked, millions in crypto stolen", "summary": "bearish crash fears"},
]
events_gdelt = [
    {"ts": now.isoformat(), "title": "Regulators signal crypto adoption push", "summary": ""},
]
events_whale = [
    {"ts": now.isoformat(), "symbol": "BTC", "amount_usd": 12_000_000},  # no title/summary -> must be excluded from scoring text
]

with open(rss_path, "w", encoding="utf-8") as f:
    for e in events_rss:
        f.write(json.dumps(e) + "\n")
with open(gdelt_path, "w", encoding="utf-8") as f:
    for e in events_gdelt:
        f.write(json.dumps(e) + "\n")
with open(whale_path, "w", encoding="utf-8") as f:
    for e in events_whale:
        f.write(json.dumps(e) + "\n")

# Force the lexicon fallback explicitly (no network / no model download).
scorer = ss.LexiconFallbackScorer()
assert isinstance(scorer, ss.LexiconFallbackScorer)

offsets = ss._load_offsets()
assert offsets == {}, "expected fresh offsets on first run"

row1 = ss.run_once(scorer, offsets)
assert row1 is not None, "expected a scored batch on first run"
assert row1["n_scored"] == 3, f"expected 3 text events scored (2 rss + 1 gdelt, whale excluded), got {row1['n_scored']}"
assert row1["polarity"] != 0.0, "expected nonzero net polarity from clearly pos/neg/pos headlines"
print("First run_once:", row1)

# Second run with no new events -> should return None (nothing new to score).
row2 = ss.run_once(scorer, offsets)
assert row2 is None, f"expected None on second run with no new events, got {row2}"
print("Second run_once (no new events):", row2)

# latest_sentiment_features() accessor should return row1's content.
latest = ss.latest_sentiment_features()
assert latest is not None
assert latest["n_scored"] == 3
print("latest_sentiment_features():", latest)

# mention_velocity should count the 3 text events (rss x2 + gdelt x1), not the whale event.
assert row1["mention_velocity"] == 3, f"expected mention_velocity=3, got {row1['mention_velocity']}"

# Restart resilience: fresh offsets dict loaded from disk should not re-score already-seen lines.
offsets_reloaded = ss._load_offsets()
row3 = ss.run_once(scorer, offsets_reloaded)
assert row3 is None, f"expected None after reloading offsets from disk (already scored), got {row3}"
print("Restart-resilience run_once:", row3)

print("ALL SENTIMENT_SCORER OFFLINE TESTS PASSED")
