import json
from pathlib import Path
from collections import deque

LEDGER_PATH = Path(r"D:\PrivateProjects\BTC Futures\scratch\live_hierarchical_session_1h_ledger.jsonl")

with open(LEDGER_PATH, "r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

# Simulate recent_mid_prices deque(maxlen=10)
recent_mid_prices = deque(maxlen=10)

for i, r in enumerate(rows):
    now_ms = r['ts_ms']
    mid = r['bbo_mid']
    pdir = r['primary_direction']
    
    # Check surge before adding current tick
    if pdir in (0, 2):
        p_30s = [p for ts, p in recent_mid_prices if (now_ms - ts) >= 20000]
        if p_30s:
            dir_mult = 1.0 if pdir == 2 else -1.0
            surge_in_dir_bps = dir_mult * ((mid - p_30s[0]) / p_30s[0] * 1e4)
            if i in (137, 138, 206, 207):
                print(f"Bar {r['bar_idx']} | Dir: {pdir} | mid: {mid:.2f} | p_30s[0]: {p_30s[0]:.2f} | surge: {surge_in_dir_bps:+.2f} bps | max_allowed: 10.0 bps")
    
    recent_mid_prices.append((now_ms, mid))
