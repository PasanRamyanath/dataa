import json
from pathlib import Path

with open("scratch/live_hierarchical_session_1h_ledger.jsonl", "r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

# Find bars where closed_trades_count increased:
last_count = 0
for i, r in enumerate(rows):
    cnt = r.get("closed_trades_count", 0)
    if cnt > last_count:
        last_count = cnt
        r_prev = rows[i-1] if i > 0 else r
        print(f"Trade #{cnt} closed around Bar {r['bar_idx']}:")
        print(f"  Pre-close pos: {r_prev.get('active_position')}")
        print(f"  Mid price: {r['bbo_mid']}")
