import json
import sys, os
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath('feature engineering scripts'))

from execution.champion_challenger_framework import (
    Cycle35_ReversionPureScalper,
    Cycle35_TightRangeReversionScalper,
)
from fusion.hierarchical_fusion_v3 import HierarchicalDecisionOutput

# Let's see what happens if we test different variations on Cycle 59 data
records = []
with open('scratch/archive_cycle59/live_hierarchical_session_1h_ledger.jsonl') as f:
    for line in f:
        records.append(json.loads(line))

print(f"Loaded {len(records)} bars from Cycle 59 ledger.")

# Let's inspect the actual trades taken in Cycle 59 by Challenger 3
with open('scratch/archive_cycle59/champion_challenger_final_results.json') as f:
    res = json.load(f)

c3_trades = res["competitor_trade_ledgers"]["Challenger_3"]
print("\nActual Challenger 3 Trades in Cycle 59:")
for i, t in enumerate(c3_trades):
    print(f"Trade {i+1}: Side={t['side']}, Entry={t['entry_price']}, Exit={t['exit_price']}, NetPnL={t['net_return_bps']:.2f} bps, Reason={t['exit_reason']}")
