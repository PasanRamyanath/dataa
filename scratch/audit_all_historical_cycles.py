import json
import glob
import os

print(f"{'Cycle':<10} | {'Winner':<32} | {'Trades':<6} | {'Wins':<4} | {'Losses':<6} | {'WR%':<6} | {'Net PnL':<10} | {'Avg PnL':<8}")
print("-" * 95)

for c in range(30, 56):
    arch = f"scratch/archive_cycle{c}"
    res_file = os.path.join(arch, "champion_challenger_final_results.json")
    if not os.path.exists(res_file):
        continue
    try:
        with open(res_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        winner = data.get('tournament_winner', 'N/A')
        wp = data.get('winner_profile', winner)
        trades = data.get('winner_trades', 0)
        net = data.get('winner_net_pnl_bps', 0.0)
        wr = data.get('winner_win_rate_pct', 0.0)
        wins = int(round(trades * wr / 100.0))
        losses = trades - wins
        avg_pnl = net / max(trades, 1) if trades > 0 else 0.0
        print(f"Cycle {c:<4} | {wp[:30]:<32} | {trades:<6} | {wins:<4} | {losses:<6} | {wr:<6.1f} | {net:<+10.2f} | {avg_pnl:<+8.2f}")
    except Exception as e:
        print(f"Cycle {c} error: {e}")
