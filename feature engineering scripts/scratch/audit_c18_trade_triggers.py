import json
from pathlib import Path

LEDGER_PATH = Path(r"D:\PrivateProjects\BTC Futures\scratch\live_hierarchical_session_1h_ledger.jsonl")

with open(LEDGER_PATH, "r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

print("=== FORENSIC ANALYSIS: WHY WERE 41 TACTICAL SIGNALS NOT CONVERTED TO TRADES? ===")
# Find all bars where execution_mode in ('PASSIVE_MAKER_SNIPER', 'WAIT_FOR_MICRO_SNIPER')
candidate_bars = []
for r in rows:
    mode = r['execution_mode']
    pdir = r['primary_direction']
    edge = r['expected_net_edge_bps']
    if mode in ('PASSIVE_MAKER_SNIPER', 'WAIT_FOR_MICRO_SNIPER') and pdir in (0, 2):
        candidate_bars.append(r)

print(f"Total candidate bars with valid direction & edge: {len(candidate_bars)}")

# Look at distribution of modes
pms_count = sum(1 for r in candidate_bars if r['execution_mode'] == 'PASSIVE_MAKER_SNIPER')
wfms_count = sum(1 for r in candidate_bars if r['execution_mode'] == 'WAIT_FOR_MICRO_SNIPER')
print(f"PASSIVE_MAKER_SNIPER bars: {pms_count} | WAIT_FOR_MICRO_SNIPER bars: {wfms_count}")

# Check what happened on the PASSIVE_MAKER_SNIPER bars
print("\nInspecting PASSIVE_MAKER_SNIPER bars in Cycle 18:")
for r in candidate_bars:
    if r['execution_mode'] == 'PASSIVE_MAKER_SNIPER':
        print(f"Bar {r['bar_idx']:3d} | {r['datetime'][11:19]} | Mid: {r['bbo_mid']:.2f} | Dir: {r['primary_direction']} | Edge: {r['expected_net_edge_bps']:.2f} | FSM: {r['fsm_state']}")

# Check why WAIT_FOR_MICRO_SNIPER bars didn't trigger
print("\nSample of WAIT_FOR_MICRO_SNIPER bars:")
for r in candidate_bars[:10]:
    if r['execution_mode'] == 'WAIT_FOR_MICRO_SNIPER':
        print(f"Bar {r['bar_idx']:3d} | {r['datetime'][11:19]} | Mid: {r['bbo_mid']:.2f} | Dir: {r['primary_direction']} | Edge: {r['expected_net_edge_bps']:.2f} | MicroProbs: {[round(x, 3) for x in r['micro_probs']]} | FSM: {r['fsm_state']}")
