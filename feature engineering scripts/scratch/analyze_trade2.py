import json
from pathlib import Path

LEDGER_PATH = Path(r"D:\PrivateProjects\BTC Futures\scratch\live_hierarchical_session_1h_ledger.jsonl")

with open(LEDGER_PATH, "r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

print("=== BARS AROUND TRADE 2 ENTRY (Bars 125 - 145) ===")
for r in rows[125:145]:
    b_idx = r['bar_idx']
    dt = r['datetime'][11:19]
    mid = r['bbo_mid']
    pdir = "SHORT" if r['primary_direction'] == 0 else ("LONG" if r['primary_direction'] == 1 else "FLAT")
    edge = r['expected_net_edge_bps']
    mode = r['execution_mode']
    fsm = r['fsm_state']
    meso = [round(x, 2) for x in r['meso_probs']]
    tac = [round(x, 2) for x in r['tactical_probs']]
    print(f"Bar {b_idx:3d} | {dt} | Mid: {mid:.2f} | Dir: {pdir:5s} | Edge: {edge:5.2f} | Mode: {mode:20s} | FSM: {fsm:20s} | Meso: {meso} | Tac: {tac}")

print("\n=== PRICE CHANGE OVER 30s & 60s LEADING UP TO BAR 139 ===")
p_139 = rows[138]['bbo_mid']
for delta in [1, 2, 3, 5, 10]:
    p_prev = rows[138 - delta]['bbo_mid']
    diff_bps = ((p_139 - p_prev) / p_prev) * 10000.0
    print(f"Lookback {delta*10}s (Bar {139-delta} -> 139): Mid {p_prev:.2f} -> {p_139:.2f} | Change: {diff_bps:+.2f} bps")
