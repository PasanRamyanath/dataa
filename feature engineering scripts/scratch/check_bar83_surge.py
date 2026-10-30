import json
from pathlib import Path
from collections import deque

LEDGER_PATH = Path(r"D:\PrivateProjects\BTC Futures\scratch\live_hierarchical_session_1h_ledger.jsonl")

with open(LEDGER_PATH, "r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

recent_mid_prices = deque(maxlen=36)
max_parabolic_surge_30s_bps = 10.0
max_impulse_surge_180s_bps = 15.0
max_impulse_surge_300s_bps = 18.0

for i, r in enumerate(rows):
    now_ms = r['ts_ms']
    mid = r['bbo_mid']
    pdir = r['primary_direction']
    
    if r['bar_idx'] == 83:
        # Check surge
        dir_mult = 1.0 if pdir == 2 else -1.0
        
        p_30s = [p for ts, p in recent_mid_prices if 20000 <= (now_ms - ts) <= 40000]
        s_30s = dir_mult * ((mid - p_30s[0]) / p_30s[0] * 1e4) if p_30s else 0.0
        
        p_180s = [p for ts, p in recent_mid_prices if 140000 <= (now_ms - ts) <= 200000]
        s_180s = dir_mult * ((mid - p_180s[0]) / p_180s[0] * 1e4) if p_180s else 0.0
        
        p_300s = [p for ts, p in recent_mid_prices if (now_ms - ts) >= 240000]
        s_300s = dir_mult * ((mid - p_300s[0]) / p_300s[0] * 1e4) if p_300s else 0.0
        
        print(f"Bar 83 | Dir={pdir} | mid={mid}")
        print(f"p_30s={p_30s} -> s_30s={s_30s:.2f} bps (max {max_parabolic_surge_30s_bps})")
        print(f"p_180s={p_180s} -> s_180s={s_180s:.2f} bps (max {max_impulse_surge_180s_bps})")
        print(f"p_300s={p_300s} -> s_300s={s_300s:.2f} bps (max {max_impulse_surge_300s_bps})")
        
    recent_mid_prices.append((now_ms, mid))
