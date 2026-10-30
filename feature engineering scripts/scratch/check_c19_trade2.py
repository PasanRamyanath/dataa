import json
from pathlib import Path

LEDGER_PATH = Path(r"D:\PrivateProjects\BTC Futures\scratch\live_hierarchical_session_1h_ledger.jsonl")

with open(LEDGER_PATH, "r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

print("=== INSPECTING TRADE 2 IN CYCLE 19 ===")
for i in range(1, len(rows)):
    prev_ap = rows[i-1].get('active_position')
    curr_ap = rows[i].get('active_position')
    if prev_ap is None and curr_ap is not None and rows[i]['bar_idx'] > 63:
        print(f"ENTRY at Bar {rows[i]['bar_idx']} ({rows[i]['datetime'][11:19]}): {curr_ap}")
    elif prev_ap is not None and curr_ap is None and rows[i]['bar_idx'] > 63:
        print(f"EXIT at Bar {rows[i]['bar_idx']} ({rows[i]['datetime'][11:19]}): BBO Mid={rows[i]['bbo_mid']:.2f}")
        print("Previous AP:", prev_ap)
