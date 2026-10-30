import json
from pathlib import Path
from collections import deque

LEDGER_PATH = Path(r"D:\PrivateProjects\BTC Futures\scratch\live_hierarchical_session_1h_ledger.jsonl")

with open(LEDGER_PATH, "r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

recent_mid_prices = deque(maxlen=36)
max_30s = 10.0
max_180s = 15.0
max_300s = 18.0

print("=== CHECKING SURGE ON BARS 227, 231, 232, 246, 260 ===")

for i, r in enumerate(rows):
    now_ms = r['ts_ms']
    mid = r['bbo_mid']
    pdir = r['primary_direction']
    
    if r['bar_idx'] in (227, 231, 232, 246, 260):
        dir_mult = 1.0 if pdir == 2 else -1.0
        
        p_30s = [p for ts, p in recent_mid_prices if 20000 <= (now_ms - ts) <= 40000]
        s_30s = dir_mult * ((mid - p_30s[0]) / p_30s[0] * 1e4) if p_30s else 0.0
        
        p_180s = [p for ts, p in recent_mid_prices if 140000 <= (now_ms - ts) <= 200000]
        s_180s = dir_mult * ((mid - p_180s[0]) / p_180s[0] * 1e4) if p_180s else 0.0
        
        p_300s = [p for ts, p in recent_mid_prices if (now_ms - ts) >= 240000]
        s_300s = dir_mult * ((mid - p_300s[0]) / p_300s[0] * 1e4) if p_300s else 0.0
        
        blocked_reasons = []
        if s_30s > max_30s: blocked_reasons.append(f"30s={s_30s:.2f}>{max_30s}")
        if s_180s > max_180s: blocked_reasons.append(f"180s={s_180s:.2f}>{max_180s}")
        if s_300s > max_300s: blocked_reasons.append(f"300s={s_300s:.2f}>{max_300s}")
        
        print(f"Bar {r['bar_idx']:3d} | Dir={pdir} | Mid={mid:.2f} | 30s={s_30s:+5.2f} | 180s={s_180s:+5.2f} | 300s={s_300s:+5.2f} | Blocked: {blocked_reasons}")
        
    recent_mid_prices.append((now_ms, mid))
