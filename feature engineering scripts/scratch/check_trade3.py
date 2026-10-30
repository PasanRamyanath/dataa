import json
from pathlib import Path

LEDGER_PATH = Path(r"D:\PrivateProjects\BTC Futures\scratch\live_hierarchical_session_1h_ledger.jsonl")

with open(LEDGER_PATH, "r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

print("=== SEARCHING FOR TRADE 3 ENTRY ===")
for r in rows[175:]:
    ap = r.get('active_position')
    if ap and ap.get('entry_price') == 79840.3:
        b_idx = r['bar_idx']
        dt = r['datetime'][11:19]
        mid = r['bbo_mid']
        edge = r['expected_net_edge_bps']
        pfe = ap.get('peak_favorable_excursion_bps', 0.0)
        mae = ap.get('max_adverse_excursion_bps', 0.0)
        print(f"First bar of Trade 3: Bar {b_idx} | {dt} | Mid: {mid} | Edge: {edge} | PFE: {pfe:.2f} | MAE: {mae:.2f}")
        break

latest = rows[-1]
ap = latest.get('active_position')
if ap:
    print(f"Latest Bar {latest['bar_idx']} | BBO: {latest['bbo_mid']} | Side: {ap['side']} | Entry: {ap['entry_price']} | PFE: {ap['peak_favorable_excursion_bps']:.2f} bps | MAE: {ap['max_adverse_excursion_bps']:.2f} bps")
