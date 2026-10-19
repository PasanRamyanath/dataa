"""
Outbound webhook delivery (§14.2) — batches queued live-loop tick records
and pushes them to Discord/Slack/a generic receiver every
config.WEBHOOK_BATCH_INTERVAL_SECONDS (default 5s), never one HTTP call per
tick. Every payload is stamped with a monotonically increasing sequence
number and session ID (streaming/sequencing.py). On a 429, backs off
exponentially with jitter, honoring the target's Retry-After (or, for
Slack-style JSON bodies, a `retry_after` field) exactly when present, and
keeps the message queued rather than dropping it.

Retry model: `_deliver()` retries one target synchronously, up to
WEBHOOK_MAX_RETRIES times with backoff+jitter between attempts, before
giving up on that flush cycle — this IS the "queue during backoff" behavior
the plan asks for, since the batch is never removed from memory while
backing off. If a target still fails after exhausting its retries, only
THAT target's batch is kept (per-target pending list, capped at
max_queue_size) and retried ahead of new ticks on the next flush, so a
failing target never makes the healthy ones receive a batch twice.

Never sends a real Discord/Slack/generic webhook request anywhere in this
codebase's own tests — every test monkeypatches the injected `session`
(a requests.Session-alike) so no live network call is made.

Usage:
    from streaming.webhook_dispatcher import WebhookDispatcher
    dispatcher = WebhookDispatcher()
    # inference/live_loop.py's run_forever(on_tick=dispatcher.on_tick) feeds it
    dispatcher.run_forever()   # blocking; run on a background thread

    python -m streaming.webhook_dispatcher --demo   # offline formatting preview, no network calls made
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import random
import threading
import time
from datetime import datetime, timezone

import requests

import config
from streaming.sequencing import Sequencer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("streaming.webhook_dispatcher")

DISCORD_CONTENT_MAX_CHARS = 1900  # stay under Discord's 2000-char content limit with margin
SLACK_TEXT_MAX_CHARS = 3800       # Slack's practical safe margin under its 4000-char text limit


def _summarize_record(r: dict) -> str:
    ts = r.get("ts", "?")
    direction = r.get("primary_direction")
    conf = r.get("confidence")
    n_voted = r.get("n_members_voted")
    trading = r.get("trading") or {}
    bits = [f"ts={ts}", f"dir={direction}", f"conf={conf:.3f}" if isinstance(conf, (int, float)) else "conf=None", f"n_voted={n_voted}"]
    if trading:
        bits.append(f"equity={trading.get('equity'):.2f}" if isinstance(trading.get("equity"), (int, float)) else "equity=?")
        bits.append(f"pos={trading.get('position_fraction')}")
    return " ".join(bits)


def _format_discord(batch: list[dict], session_id: str, sequence: int) -> dict:
    lines = [f"**Live signal batch** (session `{session_id[:8]}` seq {sequence}, {len(batch)} tick(s))"]
    lines += [f"`{_summarize_record(r)}`" for r in batch[-10:]]  # last 10 to stay within char budget
    content = "\n".join(lines)
    if len(content) > DISCORD_CONTENT_MAX_CHARS:
        content = content[:DISCORD_CONTENT_MAX_CHARS - 3] + "..."
    return {"content": content}


def _format_slack(batch: list[dict], session_id: str, sequence: int) -> dict:
    lines = [f"*Live signal batch* (session `{session_id[:8]}` seq {sequence}, {len(batch)} tick(s))"]
    lines += [f"`{_summarize_record(r)}`" for r in batch[-10:]]
    text = "\n".join(lines)
    if len(text) > SLACK_TEXT_MAX_CHARS:
        text = text[:SLACK_TEXT_MAX_CHARS - 3] + "..."
    return {"text": text}


def _format_generic(batch: list[dict], session_id: str, sequence: int) -> dict:
    return {
        "session_id": session_id,
        "sequence": sequence,
        "sent_at": datetime.now(timezone.utc).isoformat(),
        "count": len(batch),
        "records": batch,
    }


def _backoff_with_jitter(attempt: int) -> float:
    base = min(config.WEBHOOK_BACKOFF_MAX_SECONDS, config.WEBHOOK_BACKOFF_BASE_SECONDS * (2 ** attempt))
    return base * random.uniform(0.5, 1.5)


def _parse_retry_after(resp) -> float | None:
    header = resp.headers.get("Retry-After") if getattr(resp, "headers", None) else None
    if header is not None:
        try:
            return float(header)
        except (TypeError, ValueError):
            pass
    try:
        body = resp.json()
        if isinstance(body, dict) and "retry_after" in body:
            return float(body["retry_after"])
    except Exception:
        pass
    return None


class _Target:
    def __init__(self, name: str, url: str, formatter, auth_header: str = ""):
        self.name = name
        self.url = url
        self.formatter = formatter
        self.auth_header = auth_header


class WebhookDispatcher:
    def __init__(
        self,
        session: "requests.Session | None" = None,
        sequencer: Sequencer | None = None,
        discord_url: str | None = None,
        slack_url: str | None = None,
        generic_url: str | None = None,
        generic_auth_header: str | None = None,
        max_queue_size: int | None = None,
    ):
        self.session = session or requests.Session()
        self.sequencer = sequencer or Sequencer()
        self.max_queue_size = max_queue_size if max_queue_size is not None else config.WEBHOOK_MAX_QUEUE_SIZE
        self._queue: collections.deque[dict] = collections.deque()
        # on_tick runs on the live-loop thread while flush runs on the
        # dispatcher thread; the lock keeps a tick arriving mid-flush from
        # being lost between reading and clearing the queue.
        self._lock = threading.Lock()
        # Records a target failed to receive, retried ahead of new ticks on
        # the next flush — per target, so one failing target never makes
        # the others receive the same batch twice.
        self._pending: dict[str, list[dict]] = {}

        targets: list[_Target] = []
        d = discord_url if discord_url is not None else config.DISCORD_WEBHOOK_URL
        s = slack_url if slack_url is not None else config.SLACK_WEBHOOK_URL
        g = generic_url if generic_url is not None else config.GENERIC_WEBHOOK_URL
        ga = generic_auth_header if generic_auth_header is not None else config.GENERIC_WEBHOOK_AUTH_HEADER
        if d:
            targets.append(_Target("discord", d, _format_discord))
        if s:
            targets.append(_Target("slack", s, _format_slack))
        if g:
            targets.append(_Target("generic", g, _format_generic, ga))
        self.targets = targets
        if not self.targets:
            log.info("No webhook targets configured — dispatcher will queue records but deliver nothing")

    def on_tick(self, record: dict) -> None:
        """Matches inference/live_loop.py's run_forever(on_tick=...) callback
        signature exactly — pass dispatcher.on_tick directly as that hook."""
        self.enqueue(record)

    def enqueue(self, record: dict) -> None:
        with self._lock:
            self._queue.append(record)
            while len(self._queue) > self.max_queue_size:
                dropped = self._queue.popleft()
                log.warning("Webhook queue overflow (>%d) — dropped oldest record ts=%s", self.max_queue_size, dropped.get("ts"))

    def _headers_for(self, target: _Target) -> dict:
        headers = {"Content-Type": "application/json"}
        if target.auth_header:
            key, _, value = target.auth_header.partition(":")
            if key and value:
                headers[key.strip()] = value.strip()
        return headers

    def _deliver(self, target: _Target, batch: list[dict], session_id: str, sequence: int) -> bool:
        payload = target.formatter(batch, session_id, sequence)
        headers = self._headers_for(target)
        for attempt in range(config.WEBHOOK_MAX_RETRIES):
            try:
                resp = self.session.post(target.url, json=payload, headers=headers, timeout=10)
            except requests.RequestException as e:
                sleep_s = _backoff_with_jitter(attempt)
                log.warning("%s webhook request failed (%s) — retry %d/%d in %.1fs",
                            target.name, e, attempt + 1, config.WEBHOOK_MAX_RETRIES, sleep_s)
                time.sleep(sleep_s)
                continue

            if resp.status_code == 429:
                retry_after = _parse_retry_after(resp)
                sleep_s = retry_after if retry_after is not None else _backoff_with_jitter(attempt)
                log.warning("%s webhook rate-limited (429) — sleeping %.1fs before retry %d/%d",
                            target.name, sleep_s, attempt + 1, config.WEBHOOK_MAX_RETRIES)
                time.sleep(sleep_s)
                continue

            if 200 <= resp.status_code < 300:
                return True

            log.warning("%s webhook returned status %d — retry %d/%d", target.name, resp.status_code,
                        attempt + 1, config.WEBHOOK_MAX_RETRIES)
            time.sleep(_backoff_with_jitter(attempt))

        log.error("%s webhook delivery failed after %d attempts — batch will be requeued", target.name, config.WEBHOOK_MAX_RETRIES)
        return False

    def flush(self) -> dict:
        """Sends everything currently queued (plus each target's own
        previously-failed records) to every configured target and returns a
        small summary dict. Safe to call with an empty queue or no
        configured targets (both are no-ops). Every delivery is stamped with
        its own (session_id, sequence) from the Sequencer (§14.2)."""
        with self._lock:
            batch = list(self._queue)
            self._queue.clear()

        if not self.targets:
            return {"batch_size": len(batch), "targets_ok": [], "targets_failed": []}
        if not batch and not any(self._pending.values()):
            return {"batch_size": 0, "targets_ok": [], "targets_failed": []}

        ok, failed = [], []
        for target in self.targets:
            target_batch = self._pending.pop(target.name, []) + batch
            if not target_batch:
                continue
            if len(target_batch) > self.max_queue_size:
                log.warning("%s webhook backlog over %d — dropping %d oldest record(s)", target.name,
                            self.max_queue_size, len(target_batch) - self.max_queue_size)
                target_batch = target_batch[-self.max_queue_size:]
            session_id, sequence = self.sequencer.next()
            if self._deliver(target, target_batch, session_id, sequence):
                ok.append(target.name)
            else:
                failed.append(target.name)
                self._pending[target.name] = target_batch

        log.info("Flush: batch_size=%d targets_ok=%s targets_failed=%s", len(batch), ok, failed)
        return {"batch_size": len(batch), "targets_ok": ok, "targets_failed": failed}

    def run_forever(self, poll_interval: float | None = None, max_cycles: int | None = None) -> None:
        interval = poll_interval if poll_interval is not None else config.WEBHOOK_BATCH_INTERVAL_SECONDS
        log.info("Webhook dispatcher starting: %d target(s), batch every %.1fs", len(self.targets), interval)
        cycles = 0
        while max_cycles is None or cycles < max_cycles:
            start = time.monotonic()
            try:
                self.flush()
            except Exception:
                log.exception("Unhandled error in flush — continuing")
            cycles += 1
            elapsed = time.monotonic() - start
            time.sleep(max(0.0, interval - elapsed))


def _demo() -> None:
    """Offline preview: builds synthetic tick records, enqueues them, and
    prints what each formatter WOULD send — no webhook URLs configured by
    default, so flush() delivers nothing and makes no network call."""
    dispatcher = WebhookDispatcher()
    for i in range(3):
        dispatcher.on_tick({
            "ts": f"2026-01-01T00:0{i}:00", "primary_direction": i % 3, "confidence": 0.1 * i,
            "n_members_voted": 3, "trading": {"equity": 100000.0 + i * 10, "position_fraction": 0.1 * i, "trade": None},
        })
    session_id = dispatcher.sequencer.session_id
    batch = list(dispatcher._queue)
    print("Discord payload preview:", json.dumps(_format_discord(batch, session_id, 0), indent=2))
    print("Slack payload preview:", json.dumps(_format_slack(batch, session_id, 0), indent=2))
    print("Generic payload preview:", json.dumps(_format_generic(batch, session_id, 0), indent=2, default=str))
    result = dispatcher.flush()
    print("flush() result (no targets configured -> delivers nothing):", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="offline formatting preview, no network calls")
    args = parser.parse_args()
    if args.demo:
        _demo()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
