import json
from pathlib import Path

LEDGER_PATH = Path(r"D:\PrivateProjects\BTC Futures\scratch\live_hierarchical_session_1h_ledger.jsonl")

with open(LEDGER_PATH, "r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

print("=== PRICE PATH AFTER BAR 208 ===")
p_208 = rows[207]['bbo_mid']
for b in range(207, 315, 6): # every minute
    r = rows[b]
    mid = r['bbo_mid']
    diff = ((mid - p_208) / p_208) * 10000.0
    macro = [round(x, 2) for x in r['macro_probs']]
    meso = [round(x, 2) for x in r['meso_probs']]
    tac = [round(x, 2) for x in r['tactical_probs']]
    print(f"Bar {r['bar_idx']:3d} | {r['datetime'][11:19]} | Mid: {mid:.2f} | Diff: {diff:+.2f} bps | Macro: {macro} | Meso: {meso} | Tac: {tac}")
