import json
from collections import deque

with open("scratch/archive_cycle13/live_hierarchical_session_1h_ledger.jsonl", "r", encoding="utf-8") as f:
    bars = [json.loads(line) for line in f]

recent_mids = deque(maxlen=10)

print("=== BARS 118 TO 128 (CYCLE 13 SPIKE) WITH PARABOLIC GUARD ===")
for b in bars[117:128]:
    idx = b["bar_idx"]
    now_ms = b["ts_ms"]
    mid = b["bbo_mid"]
    recent_mids.append((now_ms, mid))
    
    # Check 30s surge
    p_30s = [p for ts, p in recent_mids if (now_ms - ts) >= 20000]
    surge_30s = ((mid - p_30s[0]) / p_30s[0] * 10000.0) if p_30s else 0.0
    
    dir_code = b["primary_direction"] # 2=LONG, 0=SHORT
    dir_mult = 1.0 if dir_code == 2 else (-1.0 if dir_code == 0 else 0.0)
    surge_in_dir = dir_mult * surge_30s
    is_parabolic = surge_in_dir > 10.0
    
    print(f"Bar {idx:03d} | Mid: {mid:.2f} | Dir: {dir_code} | 30s Surge: {surge_in_dir:+.2f} bps | ParabolicVeto: {is_parabolic}")
