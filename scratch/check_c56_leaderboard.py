import json

with open('scratch/champion_challenger_leaderboard.json') as f:
    data = json.load(f)

print('Timestamp:', data.get('timestamp'))
print('Bars:', data.get('bars_evaluated'))
for r in data.get('leaderboard', []):
    prof = str(r.get('strategy_profile', ''))[:60]
    print(f"Rank {r.get('rank')}: {r.get('name')} | Trades: {r.get('trades')} | Wins: {r.get('wins')} | Losses: {r.get('losses')} | Net: {r.get('net_pnl_bps')} bps | WR: {r.get('win_rate_pct')}% | Profile: {prof}")
