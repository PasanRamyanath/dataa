import json
from dataclasses import dataclass
from typing import List, Optional

# Load Cycle 11 bars
with open("scratch/archive_cycle11/live_hierarchical_session_1h_ledger.jsonl", "r", encoding="utf-8") as f:
    bars = [json.loads(line) for line in f]

print(f"Loaded {len(bars)} bars from Cycle 11 archive.")

# Inspect what happens if macro_conflict requires abs(d_macro) > 0.01 instead of 0.05
for b in bars:
    idx = b["bar_idx"]
    macro = b["macro_probs"]
    meso = b["meso_probs"]
    d_macro = macro[2] - macro[0]
    d_meso = meso[2] - meso[0]
    
    # Old conflict:
    old_conflict = (d_macro * d_meso < -0.01) and (abs(d_macro) > 0.05 and abs(d_meso) > 0.05)
    # Refined conflict:
    refined_conflict = (d_macro * d_meso < 0.0) and (abs(d_macro) > 0.015 and abs(d_meso) > 0.05)
    
    if idx in [172, 221]:
        print(f"Bar {idx}: d_macro={d_macro:+.4f}, d_meso={d_meso:+.4f} | OldConflict={old_conflict} | RefinedConflict={refined_conflict}")
