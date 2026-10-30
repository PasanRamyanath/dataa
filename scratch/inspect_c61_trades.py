import os
import sys

sys.path.insert(0, os.path.abspath('feature engineering scripts'))
sys.path.insert(0, os.path.abspath('scratch'))
from simulate_all_59_cycles_qualification import replay_cycle_ledger
from test_certified_peak_engine import Cycle64_TruePeakEngine

eng = Cycle64_TruePeakEngine(symbol="BTCUSDT")
trades = replay_cycle_ledger(eng, "scratch/archive_cycle61/live_hierarchical_session_1h_ledger.jsonl")

print(f"Cycle 61 Trades with True Peak Engine ({len(trades)} trades):")
for i, t in enumerate(trades, 1):
    print(f"  Trade {i}: {t['side']} @ {t['entry_price']:.1f} -> {t['exit_price']:.1f} | Net: {t['net_return_bps']:+6.2f} bps | Gross: {t['gross_return_bps']:+6.2f} bps | Fee: {t['total_fee_bps']:4.1f}b | Reason: {t['exit_reason']}")
