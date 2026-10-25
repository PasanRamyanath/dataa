import sys
import socket

# Apply IPv4 monkeypatch
old_getaddrinfo = socket.getaddrinfo
def new_getaddrinfo(*args, **kwargs):
    responses = old_getaddrinfo(*args, **kwargs)
    return [r for r in responses if r[0] == socket.AF_INET] or responses
socket.getaddrinfo = new_getaddrinfo

import urllib.request
import json
import time

print("Testing Binance REST connectivity with IPv4 monkeypatch...")
t0 = time.time()
req = urllib.request.Request("https://fapi.binance.com/fapi/v1/ping", headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req, timeout=5) as resp:
    res = resp.read()
    print(f"Ping Binance FAPI OK in {time.time()-t0:.2f}s: {res}")

t0 = time.time()
req2 = urllib.request.Request("https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=1m&limit=5", headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req2, timeout=5) as resp:
    data = json.loads(resp.read().decode('utf-8'))
    print(f"Fetch BTCUSDT klines OK in {time.time()-t0:.2f}s: count={len(data)}, latest close={data[-1][4]}")
