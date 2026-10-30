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

wins_data = []
losses_data = []

for l in all_ledgers:
    c_num = get_cycle_num(l)
    rep = FullCycleLedgerReplayer(l)
    eng = MacroAlignedPeakScalper(symbol="BTCUSDT")
    trades = rep.replay_engine(eng)
    for t in trades:
        record = {
            "cycle": c_num,
            "side": t.side,
            "net_pnl": t.net_return_bps,
            "mfe": t.peak_mfe_bps,
            "mae": t.max_mae_bps,
            "reason": t.exit_reason
        }
        if t.net_return_bps > 0:
            wins_data.append(record)
        else:
            losses_data.append(record)

print(f"Total Wins: {len(wins_data)} | Total Losses: {len(losses_data)}")

# Let's inspect side distribution
print("\nSide distribution:")
win_buys = sum(1 for t in wins_data if t["side"] == "BUY")
win_sells = sum(1 for t in wins_data if t["side"] == "SELL")
loss_buys = sum(1 for t in losses_data if t["side"] == "BUY")
loss_sells = sum(1 for t in losses_data if t["side"] == "SELL")

print(f"Wins: {win_buys} BUYs, {win_sells} SELLs")
print(f"Losses: {loss_buys} BUYs, {loss_sells} SELLs")
print(f"BUY Win Rate:  {win_buys / (win_buys + loss_buys) * 100:.1f}% ({win_buys}W / {loss_buys}L)")
print(f"SELL Win Rate: {win_sells / (win_sells + loss_sells) * 100:.1f}% ({win_sells}W / {loss_sells}L)")
