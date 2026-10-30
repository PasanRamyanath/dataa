import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('scratch/champion_challenger_leaderboard.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

print(f"Elapsed minutes: {d.get('elapsed_minutes')} | Bars evaluated: {d.get('bars_evaluated')}")
print("=" * 95)
print(f"{'Rank':4s} | {'Competitor':14s} | {'Trades':6s} | {'Wins':4s} | {'Losses':6s} | {'Win Rate':8s} | {'Net PnL':10s} | {'Active Pos':10s} | Strategy Profile")
print("=" * 95)
for entry in d.get('leaderboard', []):
    rank = entry.get('rank')
    name = entry.get('name')
    trades = entry.get('trades')
    wins = entry.get('wins')
    losses = entry.get('losses')
    wr = entry.get('win_rate_pct')
    net = entry.get('net_pnl_bps')
    pos = entry.get('active_pos')
    prof = entry.get('strategy_profile')
    print(f"#{rank:<3d} | {name:14s} | {trades:6d} | {wins:4d} | {losses:6d} | {wr:6.1f}% | {net:+8.2f}b | {pos:10s} | {prof}")
