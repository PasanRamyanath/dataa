import os
import sys
import json

sys.path.insert(0, os.path.abspath('feature engineering scripts'))
sys.path.insert(0, os.path.abspath('scratch'))
from simulate_all_59_cycles_qualification import replay_cycle_ledger
from execution.champion_challenger_framework import Cycle35_ReversionPureScalper

eng = Cycle35_ReversionPureScalper(symbol="BTCUSDT")
trades = replay_cycle_ledger(eng, "scratch/archive_cycle60/live_hierarchical_session_1h_ledger.jsonl")

print(f"Replay Trades count: {len(trades)}")
for i, t in enumerate(trades, 1):
    print(f"  Trade {i}: {t['side']} @ {t['entry_price']:.1f} -> {t['exit_price']:.1f} | Net: {t['net_return_bps']:+6.2f} bps | Gross: {t['gross_return_bps']:+6.2f} bps | Fee: {t['total_fee_bps']:4.1f}b | Reason: {t['exit_reason']}")

print(f"\nLive Cycle 60 trades:")
with open("scratch/archive_cycle60/champion_challenger_leaderboard.json") as f:
    d = json.load(f)
    print(f"Leaderboard: {json.dumps(d, indent=2)[:1000]}")
