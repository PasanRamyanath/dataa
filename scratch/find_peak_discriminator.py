import sys, os, json
sys.path.insert(0, os.path.abspath('feature engineering scripts'))
sys.path.insert(0, os.path.abspath('.'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from scratch.exact_c35_framework import Challenger5_ConformalConfidenceFilterScalper
from testing.run_56_cycle_historical_qualification import FullCycleLedgerReplayer, get_cycle_num
import glob

all_ledgers = sorted(
    glob.glob(str(os.path.abspath("scratch/archive_cycle*/live_hierarchical_session_1h_ledger.jsonl"))),
    key=get_cycle_num
)

records = []
for l in all_ledgers:
    c_num = get_cycle_num(l)
    rep = FullCycleLedgerReplayer(l)
    eng = Challenger5_ConformalConfidenceFilterScalper(symbol="BTCUSDT")
    trades = rep.replay_engine(eng)
    for t in trades:
        records.append({
            "cycle": c_num,
            "side": t.side,
            "entry_price": t.entry_price,
            "net_pnl": t.net_return_bps,
            "mfe": t.peak_mfe_bps,
            "mae": t.max_mae_bps,
            "duration": t.holding_duration_sec,
            "is_win": t.net_return_bps > 0
        })

wins = [r for r in records if r["is_win"]]
losses = [r for r in records if not r["is_win"]]

print(f"Total: {len(records)} | Wins: {len(wins)} | Losses: {len(losses)}")
print(f"Avg Win Duration: {sum(w['duration'] for w in wins)/len(wins):.1f}s")
print(f"Avg Loss Duration: {sum(l['duration'] for l in losses)/len(losses):.1f}s")
print(f"Avg Win MFE: {sum(w['mfe'] for w in wins)/len(wins):.2f} bps")
print(f"Avg Loss MFE: {sum(l['mfe'] for l in losses)/len(losses):.2f} bps")
print(f"Avg Loss MAE: {sum(l['mae'] for l in losses)/len(losses):.2f} bps")
