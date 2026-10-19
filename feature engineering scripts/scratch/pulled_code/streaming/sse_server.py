"""
Small FastAPI app (§14.1) serving live-loop signal data at the stable
Cloudflare Tunnel hostname:
  - GET /health             liveness probe (still secret-gated, per plan)
  - GET /latest              last known tick record (JSON)
  - GET /stream              Server-Sent Events stream of new tick records

Every endpoint requires config.STREAM_SHARED_SECRET via the
config.STREAM_SHARED_SECRET_HEADER header — this is the sole access
control; §14.3 explicitly scopes this whole layer to outbound signal
distribution only, so there is deliberately no endpoint here that accepts
input, issues commands, or reaches back into the paper-trading engine.

The "latest" cache and the SSE broadcast queue are both fed by
LiveCache.on_tick(record), which is meant to be wired to
inference/live_loop.py's on_tick hook alongside (or instead of) the
webhook dispatcher's on_tick, without modifying live_loop.py itself.

Fully testable offline via FastAPI's TestClient (no real socket bind
needed) — see the Gap 6 offline test suite.

Usage:
    from streaming.sse_server import build_app, LiveCache
    cache = LiveCache()
    app = build_app(cache)
    # uvicorn.run(app, host=config.STREAM_HOST, port=config.STREAM_PORT)

    python -m streaming.sse_server   # runs uvicorn locally for manual/offline testing
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections import deque

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse

import config

log = logging.getLogger("streaming.sse_server")

SSE_QUEUE_MAXSIZE = 500
SSE_KEEPALIVE_SECONDS = 15.0


class LiveCache:
    """Holds the most recent tick record plus a small ring buffer, and fans
    new records out to every connected SSE subscriber's own asyncio.Queue."""

    def __init__(self, history_size: int = 50):
        self.latest: dict | None = None
        self.history: deque[dict] = deque(maxlen=history_size)
        self._subscribers: set[asyncio.Queue] = set()

    def on_tick(self, record: dict) -> None:
        """Matches inference/live_loop.py's run_forever(on_tick=...)
        callback signature exactly."""
        self.latest = record
        self.history.append(record)
        for q in list(self._subscribers):
            try:
                q.put_nowait(record)
            except asyncio.QueueFull:
                log.warning("SSE subscriber queue full — dropping oldest for that subscriber")
                try:
                    q.get_nowait()
                    q.put_nowait(record)
                except Exception:
                    pass

    def subscribe(self) -> "asyncio.Queue":
        q: asyncio.Queue = asyncio.Queue(maxsize=SSE_QUEUE_MAXSIZE)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: "asyncio.Queue") -> None:
        self._subscribers.discard(q)


def _check_secret(secret_header: str | None) -> None:
    expected = config.STREAM_SHARED_SECRET
    if not expected:
        # No secret configured means the layer is deliberately left open
        # (e.g. purely local/offline testing) — never silently accept a
        # blank secret if one IS actually configured.
        return
    if secret_header != expected:
        raise HTTPException(status_code=401, detail="invalid or missing stream secret")


def build_app(cache: LiveCache | None = None) -> FastAPI:
    cache = cache if cache is not None else LiveCache()
    app = FastAPI(title="live-signal-stream")
    app.state.cache = cache

    header_name = config.STREAM_SHARED_SECRET_HEADER

    @app.get("/health")
    async def health(x_stream_secret: str | None = Header(default=None, alias=header_name)):
        _check_secret(x_stream_secret)
        return {"status": "ok"}

    @app.get("/latest")
    async def latest(x_stream_secret: str | None = Header(default=None, alias=header_name)):
        _check_secret(x_stream_secret)
        if cache.latest is None:
            return {"latest": None}
        return {"latest": cache.latest}

    @app.get("/stream")
    async def stream(x_stream_secret: str | None = Header(default=None, alias=header_name)):
        _check_secret(x_stream_secret)
        q = cache.subscribe()

        async def event_gen():
            try:
                while True:
                    try:
                        record = await asyncio.wait_for(q.get(), timeout=SSE_KEEPALIVE_SECONDS)
                        yield f"data: {json.dumps(record, default=str)}\n\n"
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
            finally:
                cache.unsubscribe(q)

        return StreamingResponse(event_gen(), media_type="text/event-stream")

    return app


def main() -> None:  # pragma: no cover - manual/local run only
    import uvicorn
    app = build_app()
    uvicorn.run(app, host=config.STREAM_HOST, port=config.STREAM_PORT)


if __name__ == "__main__":
    main()
