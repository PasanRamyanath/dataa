import json
from pathlib import Path
import numpy as np

p = Path("scratch/live_hierarchical_session_ledger.jsonl")
lines = p.read_text(encoding="utf-8").strip().split("\n")
records = [json.loads(l) for l in lines]
print(f"Total bars parsed: {len(records)}")

prices = [r["bbo_mid"] for r in records]
p_start = prices[0]
p_end = prices[-1]
p_high = max(prices)
p_low = min(prices)

print(f"Price Analysis:")
print(f"  Start: ${p_start:.2f}")
print(f"  End:   ${p_end:.2f}")
print(f"  High:  ${p_high:.2f}")
print(f"  Low:   ${p_low:.2f}")
print(f"  Net Excursion: {(p_end - p_start)/p_start*1e4:+.2f} bps ({p_end - p_start:+.2f} USD)")
print(f"  Max Drawdown (Short MFE): {(p_start - p_low)/p_start*1e4:+.2f} bps ({p_start - p_low:+.2f} USD)")
print(f"  Max Runup (Short MAE):    {(p_high - p_start)/p_start*1e4:+.2f} bps ({p_high - p_start:+.2f} USD)")

modes = {}
fsms = {}
dirs = {}
edges = []
stalking_lengths = []
current_stalking = 0
resting_events = []
current_resting = 0

for r in records:
    m = r.get("execution_mode")
    f = r.get("fsm_state")
    d = r.get("primary_direction")
    e = r.get("expected_net_edge_bps")
    
    modes[m] = modes.get(m, 0) + 1
    fsms[f] = fsms.get(f, 0) + 1
    dirs[d] = dirs.get(d, 0) + 1
    edges.append(e)
    
    if f == "RESTING_ENTRY_QUEUE":
        current_resting += 1
    else:
        if current_resting > 0:
            resting_events.append(current_resting)
            current_resting = 0

if current_resting > 0:
    resting_events.append(current_resting)

print("\nDirection Distribution (0=SHORT, 1=FLAT, 2=LONG):")
for d, count in sorted(dirs.items()):
    label = {0: "SHORT", 1: "FLAT", 2: "LONG"}.get(d, "UNKNOWN")
    print(f"  {label} ({d}): {count} bars ({count/len(records)*100:.1f}%)")

print("\nExecution Mode Distribution:")
for m, count in sorted(modes.items(), key=lambda x: x[1], reverse=True):
    print(f"  {m}: {count} bars ({count/len(records)*100:.1f}%)")

print("\nFSM State Distribution:")
for f, count in sorted(fsms.items(), key=lambda x: x[1], reverse=True):
    print(f"  {f}: {count} bars ({count/len(records)*100:.1f}%)")

print(f"\nResting Queue Placements: {len(resting_events)} separate order placements")
if resting_events:
    print(f"  Average resting duration: {np.mean(resting_events)*10:.1f} seconds (Max: {max(resting_events)*10}s)")

print(f"\nExpected Net Edge:")
print(f"  Mean:   {np.mean(edges):+.2f} bps")
print(f"  Median: {np.median(edges):+.2f} bps")
print(f"  Min:    {np.min(edges):+.2f} bps")
print(f"  Max:    {np.max(edges):+.2f} bps")
print(f"  Std:    {np.std(edges):.2f} bps")
