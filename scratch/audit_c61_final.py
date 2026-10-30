import json

with open('scratch/champion_challenger_final_results.json', 'r') as f:
    d = json.load(f)

print(f"Tournament Winner: {d.get('tournament_winner')}")
print(f"Strategy Profile: {d.get('winner_profile')}")
print(f"Winner Net PnL: {d.get('winner_net_pnl_bps'):+.2f} bps | Win Rate: {d.get('winner_win_rate_pct'):.1f}% | Trades: {d.get('winner_trades')}")

print("\n--- FINAL TOURNAMENT LEADERBOARD ---")
for row in d.get('final_leaderboard', []):
    print(f"Rank {row['rank']}: {row['name']:<15} | Trades: {row['trades']} | WR: {row['win_rate_pct']:.1f}% | Net: {row['net_pnl_bps']:+.2f}b | PF: {row['profit_factor']:.2f} | Status: {row.get('status', 'ACTIVE')}")

print("\n--- COMPETITOR TRADE COUNTS ---")
for name, trades in d.get('competitor_trade_ledgers', {}).items():
    wins = sum(1 for t in trades if t.get('net_return_bps', 0) > 0)
    pnl = sum(t.get('net_return_bps', 0) for t in trades)
    print(f"{name:<15}: {len(trades)} trades | {wins}W/{len(trades)-wins}L ({wins/len(trades)*100 if trades else 0:.1f}%) | Net: {pnl:+.2f} bps")
