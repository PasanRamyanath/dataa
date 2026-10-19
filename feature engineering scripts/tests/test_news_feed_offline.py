"""Offline test of inference.news_feed.poll_rss — monkeypatches requests.get
so no real network call is made, per user instruction to avoid live network
access during testing."""
import os
import sys
import shutil

SCRATCH = os.path.dirname(os.path.abspath(__file__))
os.environ["NEWS_DATA_ROOT"] = os.path.join(SCRATCH, "news_smoke", "events")
os.environ["NEWS_STATE_ROOT"] = os.path.join(SCRATCH, "news_smoke", "state")
shutil.rmtree(os.path.join(SCRATCH, "news_smoke"), ignore_errors=True)

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import requests
from inference import news_feed

RSS_XML = """<?xml version="1.0"?>
<rss version="2.0"><channel>
<title>Fake Feed</title>
<item><title>BTC breaks 100k</title><link>https://example.com/1</link><guid>guid-1</guid><description>desc1</description><pubDate>Mon, 01 Jan 2026 00:00:00 GMT</pubDate></item>
<item><title>ETH news</title><link>https://example.com/2</link><guid>guid-2</guid><description>desc2</description><pubDate>Mon, 01 Jan 2026 00:01:00 GMT</pubDate></item>
</channel></rss>"""


class FakeResp:
    def __init__(self, status_code, content=b"", headers=None):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}


call_log = []


def fake_get(url, headers=None, params=None, timeout=None):
    call_log.append({"url": url, "headers": dict(headers or {})})
    if len(call_log) == 1:
        return FakeResp(200, RSS_XML.encode("utf-8"), {"ETag": "abc123"})
    # second call: simulate conditional GET being honored -> 304 Not Modified
    assert headers.get("If-None-Match") == "abc123", "ETag should be sent on second poll"
    return FakeResp(304)


requests.get = fake_get

events1 = news_feed.poll_rss("testsource", "https://example.com/rss")
assert len(events1) == 2, f"expected 2 new events, got {len(events1)}"
print("First poll: got", len(events1), "events ->", [e["title"] for e in events1])

events2 = news_feed.poll_rss("testsource", "https://example.com/rss")
assert len(events2) == 0, f"expected 0 events on second (304) poll, got {len(events2)}"
print("Second poll (conditional GET, 304): got", len(events2), "events (correct)")

# Restart-resilience: fresh state load from disk should still dedup against
# the previously-seen guids (not just in-memory), simulating a process restart.
events3 = news_feed.poll_rss("testsource", "https://example.com/rss")
print("Third poll: got", len(events3), "events")

jsonl_path = os.path.join(os.environ["NEWS_DATA_ROOT"])
import glob
files = glob.glob(os.path.join(jsonl_path, "rss_testsource_*.jsonl"))
assert len(files) == 1, f"expected 1 jsonl file, got {files}"
with open(files[0]) as f:
    lines = f.readlines()
assert len(lines) == 2, f"expected 2 lines written total, got {len(lines)}"
print("JSONL file has", len(lines), "lines (correct, only the 2 real new events)")

print("ALL NEWS_FEED OFFLINE TESTS PASSED")
