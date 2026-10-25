import sys
import os
import socket
import time
import logging

# IPv4 monkeypatch
old_getaddrinfo = socket.getaddrinfo
def new_getaddrinfo(*args, **kwargs):
    responses = old_getaddrinfo(*args, **kwargs)
    return [r for r in responses if r[0] == socket.AF_INET] or responses
socket.getaddrinfo = new_getaddrinfo

# UTF-8 stdout
sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = "D:/PrivateProjects/BTC Futures"
FEAT_ENGINE_DIR = "D:/PrivateProjects/BTC Futures/feature engineering scripts"
for p in [REPO_ROOT, FEAT_ENGINE_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from streaming.run_live_hierarchical_session import LiveHierarchicalTradingSession

print("Starting 35-second dry run of LiveHierarchicalTradingSession...")
session = LiveHierarchicalTradingSession(
    symbol="BTCUSDT",
    duration_minutes=0.6,  # 36 seconds
    equity_usdt=10000.0,
)
session.start()
print("Dry run completed successfully!")
