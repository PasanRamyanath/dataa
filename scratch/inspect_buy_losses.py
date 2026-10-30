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

# Collect detailed tick-level info for all BUY trades
buy_records = []
for l in all_ledgers:
    c_num = get_cycle_num(l)
    rep = FullCycleLedgerReplayer(l)
    eng = MacroAlignedPeakScalper(symbol="BTCUSDT")
    trades = rep.replay_engine(eng)
    for t in trades:
        if t.side == "BUY":
            buy_records.append({
                "cycle": c_num,
                "is_win": t.net_return_bps > 0,
                "net_pnl": t.net_return_bps,
                "entry_price": t.entry_price,
                "mfe": t.peak_mfe_bps,
                "mae": t.max_mae_bps,
            })

win_buys = [b for b in buy_records if b["is_win"]]
loss_buys = [b for b in buy_records if not b["is_win"]]

print(f"Total BUYs: {len(buy_records)} | Wins: {len(win_buys)} | Losses: {len(loss_buys)}")

# Let's inspect where in the cycle these BUYs were taken:
# Were they taken during down cycles?
loss_cycles = set(b["cycle"] for b in loss_buys)
win_cycles = set(b["cycle"] for b in win_buys)
print(f"Cycles with winning BUYs ({len(win_cycles)}): {sorted(win_cycles)}")
print(f"Cycles with losing BUYs ({len(loss_cycles)}): {sorted(loss_cycles)}")
