"""
scratch/test_pure_stage5_supervised_trades.py
Inspect what trades Stage 5 Supervised Fusion actually authorized across Cycles 25-29.
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
        
    s5_signals = []
    for b in bars:
        s5_dir = b["primary_direction"]
        edge = b["expected_net_edge_bps"]
        tact_d = b["tactical_probs"][2] - b["tactical_probs"][0]
        meso_d = b["meso_probs"][2] - b["meso_probs"][0]
        macro_d = b["macro_probs"][2] - b["macro_probs"][0]
        
        if s5_dir in (0, 2) and edge >= 5.0:
            s5_signals.append((b["bar_idx"], b["datetime"][:19], b["bbo_mid"], "BUY" if s5_dir == 2 else "SELL", edge, macro_d, meso_d, tact_d))
            
    print(f"\n==================== CYCLE {c}: {len(s5_signals)} Stage 5 Authorized Signals ====================")
    print(f"{'Bar':<5}{'Time':<20}{'Mid':<10}{'Side':<6}{'Edge':<8}{'Macro d':<10}{'Meso d':<10}{'Tact d':<10}")
    print("-" * 80)
    for s in s5_signals:
        print(f"{s[0]:<5}{s[1]:<20}{s[2]:<10.1f}{s[3]:<6}{s[4]:<8.2f}{s[5]:<+10.2f}{s[6]:<+10.2f}{s[7]:<+10.2f}")
