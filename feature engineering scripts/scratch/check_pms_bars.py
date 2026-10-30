import json
from pathlib import Path

LEDGER_PATH = Path(r"D:\PrivateProjects\BTC Futures\scratch\live_hierarchical_session_1h_ledger.jsonl")

with open(LEDGER_PATH, "r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

for idx in [226, 230, 231, 245, 259]: # 0-indexed for Bars 227, 231, 232, 246, 260
    r = rows[idx]
    print(f"--- BAR {r['bar_idx']} ({r['datetime'][11:19]}) ---")
    print("Mode:", r['execution_mode'])
    print("FSM:", r['fsm_state'])
    print("Dir:", r['primary_direction'])
    print("Edge:", r['expected_net_edge_bps'])
    print("Micro:", r['micro_probs'])
