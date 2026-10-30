import json
from pathlib import Path

cycles = ["cycle5", "cycle10", "cycle11", "cycle12", "cycle13"]

for c in cycles:
    p = Path(f"scratch/archive_{c}/live_hierarchical_session_1h_ledger.jsonl")
    if not p.exists():
        continue
    with open(p, "r", encoding="utf-8") as f:
        bars = [json.loads(line) for line in f]
        
    entries = []
    prev_pos = None
    for i, b in enumerate(bars):
        pos = b.get("active_position")
        if pos and not prev_pos:
            # calculate 3-bar price change prior to entry
            if i >= 3:
                p_now = b["bbo_mid"]
                p_prev = bars[i-3]["bbo_mid"]
                chg_30s_bps = (p_now - p_prev) / p_prev * 10000.0
            else:
                chg_30s_bps = 0.0
            entries.append((b["bar_idx"], pos["side"], pos["entry_price"], chg_30s_bps))
        prev_pos = pos
        
    print(f"=== {c.upper()} ENTRIES & 30s PRE-ENTRY VELOCITY ===")
    for e in entries:
        dir_mult = 1.0 if e[1] == "BUY" else -1.0
        surge_in_dir = dir_mult * e[3]
        print(f"  Bar {e[0]:03d} | Side: {e[1]} | Price: {e[2]} | 30s Surge in Entry Dir: {surge_in_dir:+.2f} bps")
