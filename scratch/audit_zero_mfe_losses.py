"""
scratch/audit_zero_mfe_losses.py
Inspect the exact signals on every trade that had MFE < 1.0 bps
"""
import json
from pathlib import Path

cycles = [25, 26, 27, 28, 29]

for c in cycles:
    ledger_path = Path(f"scratch/archive_cycle{c}/live_hierarchical_session_1h_ledger.jsonl")
    if not ledger_path.exists():
        continue
    with open(ledger_path) as f:
        bars = [json.loads(line) for line in f]
    
    print(f"\n--- CYCLE {c} AUDIT ---")
    for b in bars:
        mid = b["bbo_mid"]
        macro = b["macro_probs"]
        meso = b["meso_probs"]
        tact = b["tactical_probs"]
        micro = b["micro_probs"]
        d_macro = macro[2] - macro[0]
        d_meso = meso[2] - meso[0]
        d_tact = tact[2] - tact[0]
        d_micro = micro[2] - micro[0]
        
        # Check if this was one of the bad entry prices
        # e.g. 81212.8, 81054.9, 80949.4, 80969.5, 80765.6, 80740.6, 80728.1
        bad_entries = [81212.8, 81054.9, 80949.4, 80969.5, 80765.6, 80740.6, 80728.1]
        for be in bad_entries:
            if abs(mid - be) < 2.0 and d_tact > 0.10:
                print(f"Bar {b['bar_idx']:<4} {b['datetime'][:19]} Mid: {mid:.1f} | S5 Dir: {b['primary_direction']} Edge: {b['expected_net_edge_bps']:.1f} | "
                      f"Macro d: {d_macro:+.2f} | Meso d: {d_meso:+.2f} | Tact d: {d_tact:+.2f} | Micro d: {d_micro:+.2f}")
                break
