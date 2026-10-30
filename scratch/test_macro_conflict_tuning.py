import json
from pathlib import Path

def test_conflict_on_cycle(cycle_name):
    p = Path(f"scratch/archive_{cycle_name}/live_hierarchical_session_1h_ledger.jsonl")
    if not p.exists():
        return
    with open(p, "r", encoding="utf-8") as f:
        bars = [json.loads(line) for line in f]
        
    print(f"\n=== {cycle_name.upper()} ===")
    for b in bars:
        idx = b["bar_idx"]
        macro = b["macro_probs"]
        meso = b["meso_probs"]
        d_macro = macro[2] - macro[0]
        d_meso = meso[2] - meso[0]
        
        old_conflict = (d_macro * d_meso < -0.01) and (abs(d_macro) > 0.05 and abs(d_meso) > 0.05)
        # Refined conflict: requires d_macro and d_meso to not oppose each other by > 0.015
        refined_conflict = (d_macro * d_meso < -0.005) and (abs(d_macro) > 0.015 and abs(d_meso) > 0.05)
        
        # Check notable entry bars
        # Cycle 13: bar 52 (T1 sell), bar 123 (T2 buy), bar 196 (T3 buy), bar 269 (T4 sell)
        if cycle_name == "cycle13" and idx in [52, 123, 196, 269]:
            print(f"Bar {idx:03d} (C13): d_macro={d_macro:+.4f}, d_meso={d_meso:+.4f} | Old={old_conflict} | Refined={refined_conflict}")
        # Cycle 12: bar 124 (T1 sell), bar 207 (T2 sell)
        if cycle_name == "cycle12" and idx in [124, 207]:
            print(f"Bar {idx:03d} (C12): d_macro={d_macro:+.4f}, d_meso={d_meso:+.4f} | Old={old_conflict} | Refined={refined_conflict}")
        # Cycle 11: bar 172 (T1 buy), bar 221 (T2 buy)
        if cycle_name == "cycle11" and idx in [172, 221]:
            print(f"Bar {idx:03d} (C11): d_macro={d_macro:+.4f}, d_meso={d_meso:+.4f} | Old={old_conflict} | Refined={refined_conflict}")

test_conflict_on_cycle("cycle13")
test_conflict_on_cycle("cycle12")
test_conflict_on_cycle("cycle11")
