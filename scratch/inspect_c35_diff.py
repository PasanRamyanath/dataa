import sys, os, json
sys.path.insert(0, os.path.abspath('feature engineering scripts'))
sys.path.insert(0, os.path.abspath('.'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from scratch.test_high_fidelity_replayer_v2 import HighFidelityReplayerV2
from scratch.test_strict_primary_direction import StrictPrimaryDirectionScalper

path35 = "scratch/archive_cycle35/live_hierarchical_session_1h_ledger.jsonl"
replayer = HighFidelityReplayerV2(path35)
eng = StrictPrimaryDirectionScalper(symbol="BTCUSDT")
trades = replayer.replay_engine(eng)

print(f"Cycle 35 Trades in StrictPrimaryDirectionScalper ({len(trades)} trades):")
for j, t in enumerate(trades):
    print(f"Trade {j+1}: {t.side} @ {t.entry_price:.1f} -> {t.exit_price:.1f} | Net: {t.net_return_bps:+.2f} bps | Reason: {t.exit_reason}")

print("\nActual Live Challenger 5 Trades in Cycle 35:")
print("  Trade 1: BUY @ 81175.8 -> 81368.0 | Net: +19.68 bps | PASSIVE_MAKER_TIME_UNWIND")
print("  Trade 2: BUY @ 81424.8 -> 81660.6 | Net: +24.96 bps | RESTING_MAKER_TP")
