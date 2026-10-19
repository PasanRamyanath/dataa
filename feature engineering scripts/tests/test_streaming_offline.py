"""Offline/mocked test suite for Gap 6 (streaming layer). No real Kaggle,
Cloudflare, Discord, or Slack network call is made anywhere in this file."""
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import config
from streaming.sequencing import Sequencer
from streaming import webhook_dispatcher as wd
from streaming.webhook_dispatcher import WebhookDispatcher
from streaming import cloudflared_manager as cm
from streaming.cloudflared_manager import CloudflaredManager
from streaming.sse_server import LiveCache, build_app
from streaming.live_stream_glue import build_combined_on_tick, start_background_services

tmp_root = Path(tempfile.mkdtemp(prefix="streaming_offline_"))
passed = 0
failed = 0


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"PASS: {name}")
    else:
        failed += 1
        print(f"FAIL: {name}")


# ---------------------------------------------------------------------------
# Test 1: Sequencer persistence/resumption
# ---------------------------------------------------------------------------
seq_dir = tmp_root / "seq1"
s1 = Sequencer(state_root=seq_dir)
sid1, n1 = s1.next()
sid2, n2 = s1.next()
check("sequencer increments within a process", n1 == 0 and n2 == 1 and sid1 == sid2)

s2 = Sequencer(state_root=seq_dir)  # simulate process restart, same state dir
sid3, n3 = s2.next()
check("sequencer resumes numbering across restart (same session_id, dir)", sid3 == sid1 and n3 == 2)

stamped = s2.stamp({"foo": "bar"})
check("stamp() attaches session_id/sequence", stamped["foo"] == "bar" and stamped["sequence"] == 3 and stamped["session_id"] == sid1)

corrupt_dir = tmp_root / "seq_corrupt"
corrupt_dir.mkdir(parents=True)
(corrupt_dir / "sequence_state.json").write_text("not json", encoding="utf-8")
s3 = Sequencer(state_root=corrupt_dir)
check("sequencer recovers from corrupt state file", isinstance(s3.session_id, str) and s3._next_seq == 0)


# ---------------------------------------------------------------------------
# Test 2: WebhookDispatcher batching + queue overflow
# ---------------------------------------------------------------------------
class FakeResp:
    def __init__(self, status_code=200, headers=None, body=None):
        self.status_code = status_code
        self.headers = headers or {}
        self._body = body or {}

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)  # queue of FakeResp or Exception to raise
        self.calls = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


seq_dir2 = tmp_root / "seq2"
fake_session = FakeSession([FakeResp(200), FakeResp(200)])
dispatcher = WebhookDispatcher(
    session=fake_session,
    sequencer=Sequencer(state_root=seq_dir2),
    discord_url="https://discord.example/webhook",
    slack_url="https://slack.example/webhook",
    generic_url=None,
)
dispatcher.enqueue({"ts": "t1", "primary_direction": 1, "confidence": 0.5, "n_members_voted": 2})
dispatcher.enqueue({"ts": "t2", "primary_direction": 0, "confidence": 0.1, "n_members_voted": 3})
result = dispatcher.flush()
check("flush delivers batch to both targets successfully", result["targets_ok"] == ["discord", "slack"] and result["targets_failed"] == [])
check("flush sends exactly one call per target (batched, not per-tick)", len(fake_session.calls) == 2)
check("queue is empty after a fully successful flush", len(dispatcher._queue) == 0)

empty_result = WebhookDispatcher(session=FakeSession([]), sequencer=Sequencer(state_root=tmp_root / "seq_empty")).flush()
check("flush on empty queue is a no-op", empty_result == {"batch_size": 0, "targets_ok": [], "targets_failed": []})

no_target_dispatcher = WebhookDispatcher(session=FakeSession([]), sequencer=Sequencer(state_root=tmp_root / "seq_notarget"),
                                          discord_url=None, slack_url=None, generic_url=None)
no_target_dispatcher.enqueue({"ts": "t1"})
nt_result = no_target_dispatcher.flush()
check("flush with no configured targets queues but delivers nothing", nt_result["batch_size"] == 1 and nt_result["targets_ok"] == [])

overflow_dispatcher = WebhookDispatcher(session=FakeSession([]), sequencer=Sequencer(state_root=tmp_root / "seq_overflow"),
                                         discord_url=None, slack_url=None, generic_url=None, max_queue_size=3)
for i in range(5):
    overflow_dispatcher.enqueue({"ts": f"t{i}"})
check("queue overflow drops oldest, keeps max_queue_size", len(overflow_dispatcher._queue) == 3 and overflow_dispatcher._queue[0]["ts"] == "t2")


# ---------------------------------------------------------------------------
# Test 3: 429 backoff respecting Retry-After, then requeue after exhausted retries
# ---------------------------------------------------------------------------
with mock.patch.object(wd.time, "sleep", return_value=None) as mock_sleep:
    seq_dir3 = tmp_root / "seq3"
    responses = [FakeResp(429, headers={"Retry-After": "3"})] * config.WEBHOOK_MAX_RETRIES
    fake_session2 = FakeSession(responses)
    d2 = WebhookDispatcher(session=fake_session2, sequencer=Sequencer(state_root=seq_dir3),
                            discord_url="https://discord.example/webhook", slack_url=None, generic_url=None)
    d2.enqueue({"ts": "t1"})
    r2 = d2.flush()
    check("429 exhausted retries -> target reported failed", r2["targets_failed"] == ["discord"])
    check("429 retry honors Retry-After header exactly", 3.0 in [c.args[0] for c in mock_sleep.call_args_list if c.args])
    check("batch is requeued (not dropped) after exhausted retries", len(d2._pending.get("discord", [])) == 1 and d2._pending["discord"][0]["ts"] == "t1")


# ---------------------------------------------------------------------------
# Test 4: retry_after in JSON body (Slack-style) also respected
# ---------------------------------------------------------------------------
with mock.patch.object(wd.time, "sleep", return_value=None) as mock_sleep2:
    resp_with_body = FakeResp(429, headers={}, body={"retry_after": 7})
    responses = [resp_with_body] * config.WEBHOOK_MAX_RETRIES
    fake_session3 = FakeSession(responses)
    d3 = WebhookDispatcher(session=fake_session3, sequencer=Sequencer(state_root=tmp_root / "seq4"),
                            discord_url=None, slack_url="https://slack.example/webhook", generic_url=None)
    d3.enqueue({"ts": "t1"})
    d3.flush()
    check("429 retry honors JSON retry_after field", 7.0 in [c.args[0] for c in mock_sleep2.call_args_list if c.args])


# ---------------------------------------------------------------------------
# Test 5: payload formatters
# ---------------------------------------------------------------------------
batch = [{"ts": "t1", "primary_direction": 1, "confidence": 0.42, "n_members_voted": 3,
          "trading": {"equity": 100100.0, "position_fraction": 0.2}}]
discord_payload = wd._format_discord(batch, "abcd1234efgh", 1)
slack_payload = wd._format_slack(batch, "abcd1234efgh", 1)
generic_payload = wd._format_generic(batch, "abcd1234efgh", 1)
check("discord formatter produces 'content' field under char limit", "content" in discord_payload and len(discord_payload["content"]) <= wd.DISCORD_CONTENT_MAX_CHARS)
check("slack formatter produces 'text' field under char limit", "text" in slack_payload and len(slack_payload["text"]) <= wd.SLACK_TEXT_MAX_CHARS)
check("generic formatter includes full records + session_id", generic_payload["session_id"] == "abcd1234efgh" and generic_payload["records"] == batch)


# ---------------------------------------------------------------------------
# Test 6: CloudflaredManager — no token, missing binary, mocked download/popen
# ---------------------------------------------------------------------------
cf_dir = tmp_root / "cloudflared"
mgr_no_token = CloudflaredManager(binary_path=cf_dir / "cloudflared", download_fn=lambda p, url=None: p, popen_fn=lambda *a, **kw: mock.Mock(pid=1, poll=lambda: None), token_fn=lambda: None)
started = mgr_no_token.start()
check("start() with no token returns False, never launches process", started is False and mgr_no_token._proc is None)

calls = {"download": 0, "popen": 0}


def fake_download(path, url=None):
    calls["download"] += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("fake binary")
    return path


class FakeProc:
    def __init__(self):
        self.pid = 4242
        self._alive = True

    def poll(self):
        return None if self._alive else 0

    def terminate(self):
        self._alive = False

    def wait(self, timeout=None):
        return 0


def fake_popen(*a, **kw):
    calls["popen"] += 1
    return FakeProc()


mgr = CloudflaredManager(binary_path=cf_dir / "cloudflared2", download_fn=fake_download, popen_fn=fake_popen, token_fn=lambda: "fake-token")
started2 = mgr.start()
check("start() with token + mocked download/popen launches process", started2 is True and calls["download"] == 1 and calls["popen"] == 1)
check("is_running() true after mocked start", mgr.is_running() is True)

started_again = mgr.start()
check("start() is idempotent when already running (no duplicate popen)", started_again is True and calls["popen"] == 1)

mgr.stop()
check("stop() clears the process handle", mgr._proc is None)

restarted = mgr.restart_if_dead()
check("restart_if_dead() relaunches after stop", restarted is True and calls["popen"] == 2)

# get_tunnel_token falls back to env var when kaggle_secrets is unavailable
with mock.patch.dict(cm.os.environ, {config.CLOUDFLARE_TUNNEL_TOKEN_ENV_FALLBACK: "env-fallback-token"}):
    token = cm.get_tunnel_token()
    check("get_tunnel_token() falls back to env var offline", token == "env-fallback-token")


# ---------------------------------------------------------------------------
# Test 7: FastAPI SSE server — shared-secret gating on all 3 endpoints
# ---------------------------------------------------------------------------
from fastapi.testclient import TestClient

with mock.patch.object(config, "STREAM_SHARED_SECRET", "topsecret"):
    cache = LiveCache()
    app = build_app(cache)
    client = TestClient(app)

    r_noauth = client.get("/latest")
    check("/latest rejects request with no secret header (401)", r_noauth.status_code == 401)

    r_badauth = client.get("/health", headers={config.STREAM_SHARED_SECRET_HEADER: "wrong"})
    check("/health rejects wrong secret (401)", r_badauth.status_code == 401)

    r_ok = client.get("/health", headers={config.STREAM_SHARED_SECRET_HEADER: "topsecret"})
    check("/health accepts correct secret (200)", r_ok.status_code == 200 and r_ok.json()["status"] == "ok")

    r_latest_empty = client.get("/latest", headers={config.STREAM_SHARED_SECRET_HEADER: "topsecret"})
    check("/latest returns None before any tick", r_latest_empty.status_code == 200 and r_latest_empty.json()["latest"] is None)

    cache.on_tick({"ts": "t1", "primary_direction": 1})
    r_latest_filled = client.get("/latest", headers={config.STREAM_SHARED_SECRET_HEADER: "topsecret"})
    check("/latest reflects the most recent on_tick record", r_latest_filled.json()["latest"]["ts"] == "t1")

    r_stream_noauth = client.get("/stream")
    check("/stream rejects request with no secret header (401)", r_stream_noauth.status_code == 401)


# ---------------------------------------------------------------------------
# Test 8: live_stream_glue — combined on_tick fan-out, degrade-gracefully behavior
# ---------------------------------------------------------------------------
cache2 = LiveCache()
fan_calls = {"dispatcher": 0}


class StubDispatcher:
    def on_tick(self, record):
        fan_calls["dispatcher"] += 1
        raise RuntimeError("simulated dispatcher failure")  # must not propagate


combined = build_combined_on_tick(StubDispatcher(), cache2)
combined({"ts": "t1"})  # should not raise despite dispatcher's exception
check("combined on_tick fans out to cache even if dispatcher raises", cache2.latest == {"ts": "t1"} and fan_calls["dispatcher"] == 1)

combined_no_dispatcher = build_combined_on_tick(None, cache2)
combined_no_dispatcher({"ts": "t2", "status": "scored"})
check("combined on_tick works with dispatcher disabled (None)", cache2.latest == {"ts": "t2", "status": "scored"})

# start_background_services: mock uvicorn.run so no real socket bind happens
cache3 = LiveCache()
seq_dir4 = tmp_root / "seq5"
dispatcher3 = WebhookDispatcher(session=FakeSession([]), sequencer=Sequencer(state_root=seq_dir4), discord_url=None, slack_url=None, generic_url=None)
uvicorn_calls = []


def fake_uvicorn_run(app, host=None, port=None, log_level=None):
    uvicorn_calls.append((host, port))


with mock.patch.object(cm.CloudflaredManager, "start", return_value=False):
    threads = start_background_services(cache3, dispatcher3, enable_tunnel=True, uvicorn_run_fn=fake_uvicorn_run)
    time.sleep(0.3)  # let daemon threads run their startup path
    check("start_background_services spins up expected thread count (server+dispatcher+tunnel)", len(threads) == 3)
    check("SSE server thread invoked the (mocked) uvicorn.run exactly once", len(uvicorn_calls) == 1)
    check("all started threads are daemon threads (won't block process exit)", all(t.daemon for t in threads))


# ---------------------------------------------------------------------------
# Summary + cleanup
# ---------------------------------------------------------------------------
shutil.rmtree(tmp_root, ignore_errors=True)

print(f"\n{passed} passed, {failed} failed")
if failed:
    raise SystemExit(1)
print("ALL STREAMING OFFLINE TESTS PASSED")
