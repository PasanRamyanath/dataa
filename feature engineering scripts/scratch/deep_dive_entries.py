import json
from pathlib import Path

LEDGER_PATH = Path(r"D:\PrivateProjects\BTC Futures\scratch\live_hierarchical_session_1h_ledger.jsonl")

with open(LEDGER_PATH, "r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

print("=== DEEP DIVE: TRADE 2 ENTRY (Around Bar 138) ===")
# Look at 5-minute price trajectory before Bar 138
p_entry = rows[138]['bbo_mid']
print(f"Trade 2 Entry Price: {p_entry}")
for lookback in [6, 12, 18, 30]: # 1m, 2m, 3m, 5m
    p_past = rows[138 - lookback]['bbo_mid']
    diff = ((p_entry - p_past) / p_past) * 10000.0
    print(f"Price {lookback*10}s prior (Bar {138-lookback}): {p_past:.2f} -> Move to Entry: {diff:+.2f} bps")

print("\n=== DEEP DIVE: TRADE 3 ENTRY (Around Bar 207) ===")
p_entry3 = rows[207]['bbo_mid']
print(f"Trade 3 Entry Price: {p_entry3}")
for lookback in [6, 12, 18, 30]: # 1m, 2m, 3m, 5m
    p_past = rows[207 - lookback]['bbo_mid']
    diff = ((p_entry3 - p_past) / p_past) * 10000.0
    print(f"Price {lookback*10}s prior (Bar {207-lookback}): {p_past:.2f} -> Move to Entry: {diff:+.2f} bps")

# Also let's inspect the Macro / Meso / Tactical probabilities at both entries
print("\n=== SIGNALS AT TRADE 2 (Bar 138) ===")
r2 = rows[137]
print("Macro:", [round(x, 4) for x in r2['macro_probs']])
print("Meso:", [round(x, 4) for x in r2['meso_probs']])
print("Tactical:", [round(x, 4) for x in r2['tactical_probs']])
print("Micro:", [round(x, 4) for x in r2['micro_probs']])
print("Primary Dir:", r2['primary_direction'], "Edge:", r2['expected_net_edge_bps'])

print("\n=== SIGNALS AT TRADE 3 (Bar 207) ===")
r3 = rows[206]
print("Macro:", [round(x, 4) for x in r3['macro_probs']])
print("Meso:", [round(x, 4) for x in r3['meso_probs']])
print("Tactical:", [round(x, 4) for x in r3['tactical_probs']])
print("Micro:", [round(x, 4) for x in r3['micro_probs']])
print("Primary Dir:", r3['primary_direction'], "Edge:", r3['expected_net_edge_bps'])
