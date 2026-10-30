import sys, os, json
sys.path.insert(0, os.path.abspath('feature engineering scripts'))
sys.path.insert(0, os.path.abspath('.'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from testing.run_56_cycle_historical_qualification import FullCycleLedgerReplayer, get_cycle_num
from scratch.test_macro_aligned_peak_scalper import MacroAlignedPeakScalper
import glob

all_ledgers = sorted(
    glob.glob(str(os.path.abspath("scratch/archive_cycle*/live_hierarchical_session_1h_ledger.jsonl"))),
    key=get_cycle_num
)

all_trades = []
for l in all_ledgers:
    c_num = get_cycle_num(l)
    rep = FullCycleLedgerReplayer(l)
    eng = MacroAlignedPeakScalper(symbol="BTCUSDT")
    trades = rep.replay_engine(eng)
    for t in trades:
        all_trades.append({
            "cycle": c_num,
            "side": t.side,
            "entry": t.entry_price,
            "exit": t.exit_price,
            "net_pnl": t.net_return_bps,
            "mfe": t.peak_mfe_bps,
            "mae": t.max_mae_bps,
            "reason": t.exit_reason
        })

print(f"Total trades across 56 cycles: {len(all_trades)}")
losses = [t for t in all_trades if t["net_pnl"] <= 0]
wins = [t for t in all_trades if t["net_pnl"] > 0]
print(f"Wins: {len(wins)}, Losses: {len(losses)}")

# Breakdown of losses by exit reason
reasons = {}
for t in losses:
    r = t["reason"]
    reasons[r] = reasons.get(r, 0) + 1

print("\nLosses by Exit Reason:")
for r, count in sorted(reasons.items(), key=lambda x: -x[1]):
    avg_loss = sum(t["net_pnl"] for t in losses if t["reason"] == r) / count
    print(f"  {r:35s}: {count:2d} trades | Avg loss: {avg_loss:+.2f} bps")

# How many losses reached >= +3.0 bps MFE?
reached_3 = [t for t in losses if t["mfe"] >= 3.0]
reached_4 = [t for t in losses if t["mfe"] >= 4.0]
reached_5 = [t for t in losses if t["mfe"] >= 5.0]
print(f"\nLosses that peaked in profit before reversing:")
print(f"  Peaked >= +3.0 bps MFE: {len(reached_3)} / {len(losses)} ({len(reached_3)/len(losses)*100:.1f}%)")
print(f"  Peaked >= +4.0 bps MFE: {len(reached_4)} / {len(losses)} ({len(reached_4)/len(losses)*100:.1f}%)")
print(f"  Peaked >= +5.0 bps MFE: {len(reached_5)} / {len(losses)} ({len(reached_5)/len(losses)*100:.1f}%)")
