import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

# 1. Read live results from archive_cycle62
with open('scratch/archive_cycle62/champion_challenger_final_results.json', 'r', encoding='utf-8') as f:
    live_results = json.load(f)

live_champ_trades = live_results['competitor_trade_ledgers']['Champion']
print(f"=== LIVE CYCLE 62 TRADES (Total: {len(live_champ_trades)}) ===")
for i, t in enumerate(live_champ_trades, 1):
    print(f"Live Trade {i}: {t['side']} entry={t['entry_price']} exit={t['exit_price']} net={t['net_return_bps']:+.2f}b gross={t['gross_return_bps']:+.2f}b fee={t['total_fee_bps']}b reason={t['exit_reason']}")

# 2. Run replayer on archive_cycle62
from simulate_all_59_cycles_qualification import replay_cycle_ledger
from execution.champion_challenger_framework import Cycle35_ReversionPureScalper

eng = Cycle35_ReversionPureScalper(symbol="BTCUSDT")
replayed_trades = replay_cycle_ledger(eng, "scratch/archive_cycle62/live_hierarchical_session_1h_ledger.jsonl")

print(f"\n=== REPLAYED CYCLE 62 TRADES (Total: {len(replayed_trades)}) ===")
for i, t in enumerate(replayed_trades, 1):
    print(f"Replay Trade {i}: {t['side']} entry={t['entry_price']} exit={t['exit_price']} net={t['net_return_bps']:+.2f}b gross={t['gross_return_bps']:+.2f}b fee={t['total_fee_bps']}b reason={t['exit_reason']}")
