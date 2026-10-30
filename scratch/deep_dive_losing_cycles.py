import sys, os, json
sys.path.insert(0, os.path.abspath('feature engineering scripts'))
sys.path.insert(0, os.path.abspath('.'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from scratch.test_high_fidelity_replayer_v2 import HighFidelityReplayerV2
from scratch.test_true_peak_engine_56_cycles import TruePeakDirectionalScalper

losing_cycles = ["archive_cycle44", "archive_cycle27", "archive_cycle21", "archive_cycle58"]
print("=" * 80)
print("DEEP DIVE: WHY DID THESE 4 CYCLES LOSE?")
print("=" * 80)

for c in losing_cycles:
    path = f"scratch/{c}/live_hierarchical_session_1h_ledger.jsonl"
    replayer = HighFidelityReplayerV2(path)
    eng = TruePeakDirectionalScalper(symbol="BTCUSDT")
    trades = replayer.replay_engine(eng)
    print(f"\n--- {c.upper()} ({len(trades)} trades) ---")
    for j, t in enumerate(trades):
        print(f"Trade {j+1}: {t.side} @ {t.entry_price:.1f} -> {t.exit_price:.1f} | Net: {t.net_return_bps:+.2f} bps | Peak MFE: {t.peak_mfe_bps:+.2f}b | Max MAE: {t.max_mae_bps:+.2f}b | Reason: {t.exit_reason}")
