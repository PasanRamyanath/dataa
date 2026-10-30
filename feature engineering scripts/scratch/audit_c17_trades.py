import json
from pathlib import Path

LEDGER_PATH = Path(r"D:\PrivateProjects\BTC Futures\scratch\live_hierarchical_session_1h_ledger.jsonl")

with open(LEDGER_PATH, "r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

print(f"Total bars in Cycle 17 ledger: {len(rows)}")

# Find all stalking and execution modes
stalking_bars = []
modes_seen = {}
dirs_seen = {0: 0, 1: 0, 2: 0}

for r in rows:
    pdir = r['primary_direction']
    dirs_seen[pdir] = dirs_seen.get(pdir, 0) + 1
    mode = r['execution_mode']
    modes_seen[mode] = modes_seen.get(mode, 0) + 1
    if r['fsm_state'] == 'SNIPER_ENTRY_STALKING':
        stalking_bars.append((r['bar_idx'], r['datetime'][11:19], r['bbo_mid'], pdir, r['expected_net_edge_bps'], mode))

print(f"\nDirection distribution: SHORT(0)={dirs_seen[0]}, FLAT(1)={dirs_seen[1]}, LONG(2)={dirs_seen[2]}")
print(f"Modes distribution: {modes_seen}")
print(f"\nStalking bars count: {len(stalking_bars)}")
print("\nSample of Stalking Bars (first 10 and last 10):")
for s in stalking_bars[:10]:
    print(s)
print("...")
for s in stalking_bars[-10:]:
    print(s)
