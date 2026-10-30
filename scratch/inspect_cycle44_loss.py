import json
import sys, io
sys.path.append("D:/PrivateProjects/BTC Futures/scratch")
from replay_all_64_cycles_unified_peak import engine

res = engine.replay("D:/PrivateProjects/BTC Futures/scratch/archive_cycle44/live_hierarchical_session_1h_ledger.jsonl")
print(f"Cycle 44 trades count: {len(res['trades_list'])}, Win Rate: {res['win_rate']:.1f}%, PnL: {res['pnl']:+.2f} bps")
for i, t in enumerate(res['trades_list']):
    print(f"  #{i+1}: {t['side']} Net: {t['net']:+.2f}b | MFE: {t['mfe']:.2f}b | MAE: {t['mae']:.2f}b | Dur: {t['dur']:.0f}s | Reason: {t['reason']}")
