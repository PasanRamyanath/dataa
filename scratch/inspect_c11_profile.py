import json

with open("scratch/archive_cycle11/live_hierarchical_session_1h_ledger.jsonl", "r", encoding="utf-8") as f:
    bars = [json.loads(line) for line in f]

print("=== MACRO & MESO PROBABILITIES PROFILE ===")
macro_deltas = [b["macro_probs"][2] - b["macro_probs"][0] for b in bars]
meso_deltas = [b["meso_probs"][2] - b["meso_probs"][0] for b in bars]

print(f"Macro delta: min={min(macro_deltas):.4f}, max={max(macro_deltas):.4f}, mean={sum(macro_deltas)/len(macro_deltas):.4f}")
print(f"Meso delta:  min={min(meso_deltas):.4f}, max={max(meso_deltas):.4f}, mean={sum(meso_deltas)/len(meso_deltas):.4f}")

# Count bars by primary direction
from collections import Counter
c = Counter(b["primary_direction"] for b in bars)
print("Primary direction counts:", { "SHORT(0)": c[0], "FLAT(1)": c[1], "LONG(2)": c[2] })
