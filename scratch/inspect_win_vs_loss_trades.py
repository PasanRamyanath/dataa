import sys, os, json
sys.path.insert(0, os.path.abspath('feature engineering scripts'))
sys.path.insert(0, os.path.abspath('.'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from testing.run_56_cycle_historical_qualification import FullCycleLedgerReplayer
from scratch.test_macro_aligned_peak_scalper import MacroAlignedPeakScalper

for c_num in [35, 48, 57]:
    path = f"scratch/archive_cycle{c_num}/live_hierarchical_session_1h_ledger.jsonl"
    rep = FullCycleLedgerReplayer(path)
    eng = MacroAlignedPeakScalper(symbol="BTCUSDT")
    trades = rep.replay_engine(eng)
    print(f"\n=================== CYCLE {c_num} TRADES ({len(trades)}) ===================")
    for j, t in enumerate(trades):
        print(f"Trade {j+1}: {t.side} @ {t.entry_price:.1f} -> {t.exit_price:.1f} | Net: {t.net_return_bps:+.2f} bps | Peak MFE: {t.peak_mfe_bps:+.2f}b | Max MAE: {t.max_mae_bps:+.2f}b | Reason: {t.exit_reason}")
