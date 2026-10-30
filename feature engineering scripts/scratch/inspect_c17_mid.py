import json
from pathlib import Path

LEDGER_PATH = Path(r"D:\PrivateProjects\BTC Futures\scratch\live_hierarchical_session_1h_ledger.jsonl")

with open(LEDGER_PATH, "r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

print("=== BARS 90 TO 180 SUMMARY ===")
for r in rows[90:180:10]: # every 10 bars (100 seconds)
    b_idx = r['bar_idx']
    dt = r['datetime'][11:19]
    mid = r['bbo_mid']
    pdir = "SHORT" if r['primary_direction'] == 0 else ("LONG" if r['primary_direction'] == 2 else "FLAT")
    edge = r['expected_net_edge_bps']
    mode = r['execution_mode']
    fsm = r['fsm_state']
    macro = [round(x, 2) for x in r['macro_probs']]
    meso = [round(x, 2) for x in r['meso_probs']]
    tac = [round(x, 2) for x in r['tactical_probs']]
    print(f"Bar {b_idx:3d} | {dt} | Mid: {mid:.2f} | Dir: {pdir:5s} | Edge: {edge:5.2f} | Mode: {mode:20s} | FSM: {fsm:20s} | Meso: {meso} | Tac: {tac}")
