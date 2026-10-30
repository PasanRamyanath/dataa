import json
with open('scratch/champion_challenger_leaderboard.json') as f:
    d = json.load(f)
for c in d.get('leaderboard', []):
    print(f"Rank {c['rank']}: {c['name']} | Trades: {c['trades']} | Net: {c['net_pnl_bps']} bps | WR: {c['win_rate_pct']}% | Active: {c['active_pos']}")
