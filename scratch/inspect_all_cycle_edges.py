import json
from pathlib import Path

cycles = ["cycle5", "cycle6", "cycle7", "cycle8", "cycle9", "cycle10", "cycle11"]

for c in cycles:
    p = Path(f"scratch/archive_{c}/live_hierarchical_session_1h_ledger.jsonl")
    if not p.exists():
        continue
    with open(p, "r", encoding="utf-8") as f:
        bars = [json.loads(line) for line in f]
    
    entries = []
    prev_pos = None
    for b in bars:
        pos = b.get("active_position")
        if pos and not prev_pos:
            entries.append((b["bar_idx"], pos["side"], pos["entry_price"], b.get("expected_net_edge_bps", 0.0)))
        prev_pos = pos
        
    print(f"=== {c.upper()} ({len(entries)} trade entries) ===")
    for e in entries:
        print(f"  Bar {e[0]:03d} | Side: {e[1]} | Price: {e[2]} | Entry Edge: {e[3]:+.2f} bps")
