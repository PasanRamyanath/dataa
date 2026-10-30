"""
scratch/check_c28_fusion_vs_tactical.py
Check whether Stage 5 Fusion authorized the entries in Cycle 28
"""
import json

with open("scratch/archive_cycle28/champion_challenger_final_results.json") as f:
    c28_results = json.load(f)

champ_trades = c28_results["competitor_trade_ledgers"]["Challenger_1"]

# Load Cycle 28 ledger
ledger_bars = {}
with open("scratch/archive_cycle28/live_hierarchical_session_1h_ledger.jsonl") as f:
    for line in f:
        bar = json.loads(line)
        ledger_bars[bar["ts_ms"]] = bar

print(f"{'Tr#':<4}{'Side':<5}{'Entry Time':<25}{'Entry P':<10}{'Exit P':<10}{'Net PnL':<10}{'Reason':<25}{'Stage5 Dir':<12}{'Stage5 Edge':<12}{'Meso d':<10}{'Tact d':<10}")
print("-" * 125)

for i, t in enumerate(champ_trades, 1):
    entry_ts = t["entry_ts_ms"]
    bar = ledger_bars.get(entry_ts)
    if not bar:
        closest = min(ledger_bars.keys(), key=lambda x: abs(x - entry_ts))
        bar = ledger_bars[closest]
    
    stage5_dir = bar["primary_direction"]
    stage5_edge = bar["expected_net_edge_bps"]
    tact_d = bar["tactical_probs"][2] - bar["tactical_probs"][0]
    meso_d = bar["meso_probs"][2] - bar["meso_probs"][0]
    
    dir_str = "FLAT(1)" if stage5_dir == 1 else ("BUY(2)" if stage5_dir == 2 else "SELL(0)")
    print(f"{i:<4}{t['side']:<5}{bar['datetime'][:19]:<25}{t['entry_price']:<10.1f}{t['exit_price']:<10.1f}{t['net_return_bps']:<+10.2f}{t['exit_reason']:<25}{dir_str:<12}{stage5_edge:<12.2f}{meso_d:<+10.2f}{tact_d:<+10.2f}")
