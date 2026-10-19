"""
Event-triggered sentiment scoring per plan §9.2/§9.3.

Watches the JSONL event files inference/news_feed.py writes to
config.NEWS_DATA_ROOT and, whenever new text-bearing events (RSS headlines,
GDELT articles) show up, scores them in a batch with a quantized FinBERT-class
classifier (config.SENTIMENT_MODEL_NAME — well under 500MB at int8/fp16 per
§7.1) for polarity/magnitude/confidence, plus a mention-velocity counter
(story volume in a trailing window, config.SENTIMENT_MENTION_VELOCITY_WINDOW_SECONDS).
Per §9.2, this is explicitly event-triggered batch scoring, NOT a per-tick
(1Hz) general-purpose LLM call — an unoptimized 7B model on a T4 only manages
~4 tokens/sec, and that GPU headroom is needed by the price-prediction
members.

"Event-triggered" here means: this process polls the day's JSONL files for
new lines every --poll-interval seconds (default 5s) and only runs the model
when it actually finds new lines — filesystem watching (inotify/watchdog)
would be more elegant but adds a new dependency for a marginal latency win
over a 5s poll, given headline sentiment itself is only useful on a
minutes-to-hours drift timescale anyway (see plan §9's own framing).

Graceful degradation: if `transformers`/`torch` aren't installed, or the
model can't be downloaded (no network, no HF access), this falls back to a
small hand-built lexicon polarity heuristic and logs a clear one-time
warning — the pipeline still produces a (lower-quality but real) sentiment
signal rather than crashing, matching this repo's established
graceful-degradation conventions (models/train_all.py, common/hf_model_upload.py).

Output: appends (ts, polarity, magnitude, confidence, mention_velocity,
n_scored) snapshot rows to config.SENTIMENT_FEATURES_ROOT/{date}.jsonl every
time a batch is scored. latest_sentiment_features() reads the most recent
row back — the accessor inference/live_loop.py uses to fetch the current
sentiment state at inference time. The training-time path — full-text
scraping with historical backfill, two-tier (this classifier + an instruct
LLM) scoring, and point-in-time multi-window features the models actually
train on — is news_pipeline/, which reuses load_scorer() below as its
tier 1.

Usage:
    python -m inference.sentiment_scorer              # run forever
    python -m inference.sentiment_scorer --once        # one scoring pass and exit (testing)
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("inference.sentiment_scorer")

TEXT_SOURCES_PREFIX = ("rss_", "gdelt")  # whale_alert events carry no text to score
SCORE_BATCH_SIZE = 64  # bounds GPU memory when a restart finds a large backlog


def _recent_text_files(now: datetime | None = None) -> list[Path]:
    """Today's + yesterday's text-source JSONL files only (news_feed names
    them {source}_{YYYY-MM-DD}.jsonl) — never the whole growing history."""
    if not config.NEWS_DATA_ROOT.exists():
        return []
    now = now or datetime.now(timezone.utc)
    days = {now.strftime("%Y-%m-%d"), (now - timedelta(days=1)).strftime("%Y-%m-%d")}
    return sorted(p for p in config.NEWS_DATA_ROOT.glob("*.jsonl")
                  if p.name.startswith(TEXT_SOURCES_PREFIX) and p.stem[-10:] in days)

_POSITIVE_WORDS = {
    "surge", "rally", "bullish", "gain", "gains", "soar", "soars", "record", "high", "breakout",
    "approval", "approved", "adoption", "inflow", "inflows", "buy", "accumulate", "upgrade", "win",
}
_NEGATIVE_WORDS = {
    "crash", "plunge", "bearish", "loss", "losses", "sell-off", "selloff", "dump", "hack", "hacked",
    "ban", "banned", "lawsuit", "collapse", "liquidation", "liquidated", "outflow", "outflows", "fear",
    "reject", "rejected", "downgrade", "exploit", "scam",
}


class LexiconFallbackScorer:
    """Deterministic, dependency-free polarity heuristic used only when the
    real FinBERT-class model can't be loaded — a documented degraded mode,
    not a silent substitute (see module docstring)."""

    def score_batch(self, texts: list[str]) -> list[dict]:
        out = []
        for text in texts:
            words = text.lower().split()
            pos = sum(1 for w in words if w.strip(".,!?:;\"'()") in _POSITIVE_WORDS)
            neg = sum(1 for w in words if w.strip(".,!?:;\"'()") in _NEGATIVE_WORDS)
            total_hits = pos + neg
            polarity = 0.0 if total_hits == 0 else (pos - neg) / total_hits
            magnitude = min(1.0, total_hits / 5.0)
            confidence = 0.3 if total_hits == 0 else min(0.6, 0.3 + 0.1 * total_hits)
            out.append({"polarity": polarity, "magnitude": magnitude, "confidence": confidence})
        return out


class FinBertScorer:
    """Wraps a quantized (int8 on CPU / fp16 on GPU) AutoModelForSequenceClassification
    (config.SENTIMENT_MODEL_NAME). Raises on construction if transformers/torch
    or the model weights aren't available — callers should catch and fall
    back to LexiconFallbackScorer (see load_scorer())."""

    def __init__(self, model_name: str):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        else:
            model = model.half().to(self.device)
        model.eval()
        self.model = model
        self.id2label = {int(k): v for k, v in model.config.id2label.items()}
        self.torch = torch

    def score_batch(self, texts: list[str]) -> list[dict]:
        torch = self.torch
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits.float()
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

        neutral_idx = [i for i, l in self.id2label.items() if "neu" in l.lower()]
        out = []
        for row in probs:
            top_idx = int(row.argmax())
            label = self.id2label.get(top_idx, str(top_idx)).lower()
            confidence = float(row[top_idx])
            if "pos" in label or "bull" in label:
                polarity = confidence
            elif "neg" in label or "bear" in label:
                polarity = -confidence
            else:
                polarity = 0.0  # neutral
            # Non-neutral probability mass; a model without a neutral class
            # has no such split, so its top-class confidence stands in.
            magnitude = float(1.0 - row[neutral_idx].sum()) if neutral_idx else confidence
            out.append({"polarity": polarity, "magnitude": max(0.0, min(1.0, magnitude)), "confidence": confidence})
        return out


def load_scorer(model_name: str | None = None):
    model_name = model_name or config.SENTIMENT_MODEL_NAME
    try:
        scorer = FinBertScorer(model_name)
        log.info("Loaded FinBERT-class scorer '%s' on %s", model_name, scorer.device)
        return scorer
    except Exception as e:
        log.warning(
            "Could not load sentiment model '%s' (%s: %s) — falling back to lexicon-based polarity heuristic. "
            "Install `transformers`/`torch` (requirements-models.txt) and ensure the model is reachable "
            "(HF Hub network access) for real FinBERT-class scoring.",
            model_name, type(e).__name__, e,
        )
        return LexiconFallbackScorer()


def _event_text(event: dict) -> str:
    title = event.get("title", "") or ""
    summary = event.get("summary", "") or ""
    return f"{title}. {summary}".strip(". ")


def _iter_new_text_events(offsets: dict[str, int]) -> list[dict]:
    """Reads every complete line beyond the previously-recorded offset in
    each of today's + yesterday's text-source JSONL files (yesterday too, so
    events written right at a UTC day rollover aren't missed). A trailing
    line without its newline is still being written by news_feed and is
    left for the next pass instead of being dropped as malformed."""
    new_events = []
    files = _recent_text_files()
    for path in files:
        key = str(path)
        start_line = offsets.get(key, 0)
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if lines and not lines[-1].endswith("\n"):
            lines = lines[:-1]
        if len(lines) <= start_line:
            continue
        for line in lines[start_line:]:
            line = line.strip()
            if not line:
                continue
            try:
                new_events.append(json.loads(line))
            except json.JSONDecodeError:
                log.warning("Skipping malformed JSONL line in %s", path)
        offsets[key] = len(lines)
    # Forget offsets of files that have aged out of the two-day window.
    live = {str(p) for p in files}
    for key in [k for k in offsets if k not in live]:
        del offsets[key]
    return new_events


def _load_offsets() -> dict[str, int]:
    path = config.NEWS_STATE_ROOT / "sentiment_scorer_offsets.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_offsets(offsets: dict[str, int]) -> None:
    config.NEWS_STATE_ROOT.mkdir(parents=True, exist_ok=True)
    (config.NEWS_STATE_ROOT / "sentiment_scorer_offsets.json").write_text(json.dumps(offsets), encoding="utf-8")


def _mention_velocity(now: datetime, window_seconds: int) -> int:
    """Counts text-source events ingested within the trailing window, across
    all of today's (+ yesterday's, for a window spanning midnight) JSONL
    files — independent of `offsets`, since mention velocity is about total
    recent story volume, not just this run's newly-scored batch."""
    cutoff = now - timedelta(seconds=window_seconds)
    count = 0
    for path in _recent_text_files(now):
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ts_str = ev.get("ts")
                    if not ts_str:
                        continue
                    ts = datetime.fromisoformat(ts_str)
                    if ts >= cutoff:
                        count += 1
        except OSError:
            continue
    return count


def _append_feature_row(row: dict) -> Path:
    config.SENTIMENT_FEATURES_ROOT.mkdir(parents=True, exist_ok=True)
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = config.SENTIMENT_FEATURES_ROOT / f"{day}.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")
    return path


def latest_sentiment_features() -> dict | None:
    """Accessor for inference/live_loop.py: the most recently scored
    sentiment snapshot, or None if the scorer hasn't produced one yet
    (graceful degradation — see inference/live_loop.py's per-member skip
    logic, the same pattern for a not-yet-available signal)."""
    if not config.SENTIMENT_FEATURES_ROOT.exists():
        return None
    files = sorted(config.SENTIMENT_FEATURES_ROOT.glob("*.jsonl"))
    for path in reversed(files):
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        if lines:
            return json.loads(lines[-1])
    return None


def run_once(scorer, offsets: dict[str, int]) -> dict | None:
    events = _iter_new_text_events(offsets)
    if not events:
        return None

    texts = [_event_text(ev) for ev in events]
    scores = []
    for i in range(0, len(texts), SCORE_BATCH_SIZE):
        scores.extend(scorer.score_batch(texts[i:i + SCORE_BATCH_SIZE]))

    now = datetime.now(timezone.utc)
    row = {
        "ts": now.isoformat(),
        "polarity": float(sum(s["polarity"] for s in scores) / len(scores)),
        "magnitude": float(sum(s["magnitude"] for s in scores) / len(scores)),
        "confidence": float(sum(s["confidence"] for s in scores) / len(scores)),
        "mention_velocity": _mention_velocity(now, config.SENTIMENT_MENTION_VELOCITY_WINDOW_SECONDS),
        "n_scored": len(events),
        "scorer": type(scorer).__name__,
    }
    _append_feature_row(row)
    _save_offsets(offsets)
    log.info("Scored %d new event(s): polarity=%.3f magnitude=%.3f confidence=%.3f mention_velocity=%d",
              len(events), row["polarity"], row["magnitude"], row["confidence"], row["mention_velocity"])
    return row


def run_forever(scorer, poll_interval: float, max_cycles: int | None = None) -> None:
    offsets = _load_offsets()
    cycles = 0
    while max_cycles is None or cycles < max_cycles:
        try:
            run_once(scorer, offsets)
        except Exception:
            log.exception("Sentiment scoring cycle failed — will retry next cycle")
        cycles += 1
        time.sleep(poll_interval)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--max-cycles", type=int, default=None)
    args = parser.parse_args()

    scorer = load_scorer(args.model_name)
    if args.once:
        offsets = _load_offsets()
        result = run_once(scorer, offsets)
        log.info("Single scoring pass complete: %s", "no new events" if result is None else result)
    else:
        run_forever(scorer, args.poll_interval, args.max_cycles)


if __name__ == "__main__":
    main()
