import json

with open('scratch/archive_cycle35/champion_challenger_final_results.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

for c in d.get('final_leaderboard', []):
    print(c.get('name'), '|', c.get('strategy_profile'), '| Net:', c.get('net_pnl_bps'), '| WR:', c.get('win_rate_pct'), '| Trades:', c.get('trades'))
