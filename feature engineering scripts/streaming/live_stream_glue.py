"""
Wires inference/live_loop.py's existing on_tick(record) hook to both the
webhook dispatcher (streaming/webhook_dispatcher.py) and the SSE "latest"
cache (streaming/sse_server.py), plus starts the named Cloudflare tunnel
(streaming/cloudflared_manager.py) and the FastAPI server in background
threads — all WITHOUT modifying inference/live_loop.py itself, exactly as
that module's own docstring says this integration should happen (§14, "a
later streaming/webhook_dispatcher.py can pass its own sink function
without live_loop.py needing to change").

This is the single process a Kaggle inference cell would actually run.
Every piece degrades gracefully and independently:
  - no webhook URLs configured -> dispatcher just queues and drops nothing,
    delivers nothing (see webhook_dispatcher.WebhookDispatcher.flush).
  - no Cloudflare tunnel token available -> tunnel is skipped, SSE server
    still runs locally (just not reachable from outside the VM).
  - the SSE server failing to start -> logged, webhook delivery and the
    live loop itself keep running regardless.
None of these failures crash the live inference loop, matching the "skip
this piece, log it, keep going" pattern used throughout this repo.

Never opens a real tunnel, sends a real webhook, or binds a real inbound
port in this codebase's own tests — the offline test suite drives
build_combined_on_tick() and start_background_services() directly with
every subprocess/network dependency mocked.

Usage:
    python -m streaming.live_stream_glue                 # run forever, full stack
    python -m streaming.live_stream_glue --no-tunnel       # skip cloudflared (local-only SSE)
    python -m streaming.live_stream_glue --no-webhooks     # skip Discord/Slack/generic delivery
    python -m streaming.live_stream_glue --once             # single tick, for smoke-testing wiring
"""
from __future__ import annotations

import argparse
import logging
import threading
import time
from typing import Callable

import config
from inference import live_loop
from streaming.cloudflared_manager import CloudflaredManager
from streaming.sequencing import Sequencer
from streaming.sse_server import LiveCache, build_app
from streaming.webhook_dispatcher import WebhookDispatcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("streaming.live_stream_glue")

TUNNEL_WATCHDOG_SECONDS = 30.0


def build_combined_on_tick(
    dispatcher: WebhookDispatcher | None,
    cache: LiveCache | None,
) -> Callable[[dict], None]:
    """Returns one callback matching live_loop.run_forever(on_tick=...)'s
    exact signature that fans a single tick record out to both sinks.
    Either sink may be None (disabled) independently."""

    def _on_tick(record: dict) -> None:
        if cache is not None:
            try:
                cache.on_tick(record)
            except Exception:
                log.exception("SSE cache on_tick failed — continuing")
        if dispatcher is not None:
            try:
                dispatcher.on_tick(record)
            except Exception:
                log.exception("Webhook dispatcher on_tick failed — continuing")

    return _on_tick


def start_background_services(
    cache: LiveCache,
    dispatcher: WebhookDispatcher | None,
    enable_tunnel: bool = True,
    uvicorn_run_fn=None,
) -> list[threading.Thread]:
    """Starts the FastAPI/SSE server, the webhook dispatcher's flush loop,
    and (optionally) the Cloudflare tunnel, each on its own daemon thread so
    a failure in one never blocks the others or the live loop. Returns the
    started threads (for tests to join/inspect, not required in normal
    operation since these are daemon threads)."""
    threads: list[threading.Thread] = []

    def _run_server():
        try:
            import uvicorn
            app = build_app(cache)
            run = uvicorn_run_fn if uvicorn_run_fn is not None else uvicorn.run
            run(app, host=config.STREAM_HOST, port=config.STREAM_PORT, log_level="warning")
        except Exception:
            log.exception("SSE/FastAPI server failed to start — /latest and /stream unavailable this session")

    t_server = threading.Thread(target=_run_server, name="sse-server", daemon=True)
    t_server.start()
    threads.append(t_server)

    if dispatcher is not None:
        t_dispatch = threading.Thread(target=dispatcher.run_forever, name="webhook-dispatcher", daemon=True)
        t_dispatch.start()
        threads.append(t_dispatch)

    if enable_tunnel:
        def _run_tunnel():
            try:
                mgr = CloudflaredManager()
                if not mgr.start():
                    return  # no token/binary: nothing to watch
            except Exception:
                log.exception("Cloudflare tunnel failed to start — stream will only be reachable locally this session")
                return
            # Watchdog: relaunch cloudflared if it dies mid-session.
            while True:
                time.sleep(TUNNEL_WATCHDOG_SECONDS)
                try:
                    mgr.restart_if_dead()
                except Exception:
                    log.exception("cloudflared restart attempt failed — will retry")

        t_tunnel = threading.Thread(target=_run_tunnel, name="cloudflared", daemon=True)
        t_tunnel.start()
        threads.append(t_tunnel)

    return threads


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="single tick, for smoke-testing wiring")
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--no-trading", action="store_true")
    parser.add_argument("--no-tunnel", action="store_true", help="skip starting the Cloudflare tunnel")
    parser.add_argument("--no-webhooks", action="store_true", help="skip webhook delivery, SSE/latest cache only")
    args = parser.parse_args()

    cache = LiveCache()
    dispatcher = None if args.no_webhooks else WebhookDispatcher(sequencer=Sequencer())
    on_tick = build_combined_on_tick(dispatcher, cache)

    if not args.once:
        start_background_services(cache, dispatcher, enable_tunnel=not args.no_tunnel)

    if args.once:
        state = live_loop.LiveLoopState(enable_trading=not args.no_trading)
        record = live_loop.run_tick(state)
        on_tick(record)
        log.info("Single tick complete: %s", record.get("ts"))
    else:
        live_loop.run_forever(
            enable_trading=not args.no_trading,
            max_cycles=args.max_cycles,
            on_tick=on_tick,
        )


if __name__ == "__main__":
    main()
