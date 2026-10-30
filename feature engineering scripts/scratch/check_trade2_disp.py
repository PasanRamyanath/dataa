import json
from pathlib import Path

LEDGER_PATH = Path(r"D:\PrivateProjects\BTC Futures\scratch\live_hierarchical_session_1h_ledger.jsonl")

with open(LEDGER_PATH, "r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

r158 = rows[157]
p_entry = r158['bbo_mid']
print(f"Trade 2 Entry at Bar 158: Mid={p_entry:.2f}")

for delta in [3, 6, 18, 30]:
    p_past = rows[157 - delta]['bbo_mid']
    # Displacement in short direction (p_past - p_entry)
    disp = ((p_past - p_entry) / p_past) * 1e4
    print(f"Lookback {delta*10}s (Bar {158-delta}): Past={p_past:.2f} -> Move to Entry: {disp:+.2f} bps")
