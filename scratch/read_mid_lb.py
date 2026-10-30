import json

with open('scratch/champion_challenger_leaderboard.json', 'r') as f:
    d = json.load(f)

print(f"Elapsed: {d.get('elapsed_minutes')} min | Bars: {d.get('bars_evaluated')} | BTC: ${d.get('current_price')}")
print(f"{'Rank':<5} | {'Competitor':<15} | {'Trades':<6} | {'Wins':<4} | {'Losses':<6} | {'Win Rate':<8} | {'Net PnL':<10} | {'PF':<5}")
print("-" * 75)
for row in d.get('leaderboard', []):
    print(f"{row['rank']:<5} | {row['name']:<15} | {row['trades']:<6} | {row['wins']:<4} | {row['losses']:<6} | {row['win_rate_pct']:>6.1f}% | {row['net_pnl_bps']:>+8.2f}b | {row['profit_factor']:>4.2f}")
