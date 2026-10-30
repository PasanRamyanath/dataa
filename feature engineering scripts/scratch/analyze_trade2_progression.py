import json
from pathlib import Path

LEDGER_PATH = Path(r"D:\PrivateProjects\BTC Futures\scratch\live_hierarchical_session_1h_ledger.jsonl")

with open(LEDGER_PATH, "r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

print("=== TRADE 2 PROGRESSION (Bars 138 to 180) ===")
for r in rows[138:181]:
    b_idx = r['bar_idx']
    dt = r['datetime'][11:19]
    mid = r['bbo_mid']
    pdir_raw = r['primary_direction']
    pdir = "SHORT" if pdir_raw == 0 else ("LONG" if pdir_raw == 2 else "FLAT")
    edge = r['expected_net_edge_bps']
    mode = r['execution_mode']
    ap = r['active_position']
    if ap:
        unreal = ((mid - ap['entry_price']) / ap['entry_price']) * 10000.0 if ap['side'] == 'BUY' else ((ap['entry_price'] - mid) / ap['entry_price']) * 10000.0
        pfe = ap.get('peak_favorable_excursion_bps', 0.0)
        mae = ap.get('max_adverse_excursion_bps', 0.0)
        print(f"Bar {b_idx:3d} | {dt} | Mid: {mid:.2f} | Dir: {pdir:5s} | Edge: {edge:5.2f} | Unreal: {unreal:+6.2f} bps | Peak: {pfe:5.2f} bps | MaxAdv: {mae:6.2f} bps")
    else:
        print(f"Bar {b_idx:3d} | {dt} | Mid: {mid:.2f} | Dir: {pdir:5s} | Edge: {edge:5.2f} | NO POSITION (FSM: {r['fsm_state']})")
