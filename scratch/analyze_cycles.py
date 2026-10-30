import os
import json
from pathlib import Path

scratch = Path("scratch")
archives = [d for d in scratch.iterdir() if d.is_dir() and d.name.startswith("archive_cycle")]
archives.sort(key=lambda x: int(x.name.replace("archive_cycle", "")))

print(f"Found {len(archives)} archive cycle directories.")
results = []

for d in archives:
    c_num = int(d.name.replace("archive_cycle", ""))
    res_file = d / "champion_challenger_final_results.json"
    if not res_file.exists():
        continue
    try:
        data = json.loads(res_file.read_text(encoding="utf-8"))
        winner = data.get("tournament_winner", "Unknown")
        winner_pnl = data.get("winner_net_pnl_bps", 0.0)
        winner_wr = data.get("winner_win_rate_pct", 0.0)
        winner_trades = data.get("winner_trades", 0)
        leaderboard = data.get("final_leaderboard", [])
        
        # also find best competitor in that cycle
        best_pnl = max([c.get("net_pnl_bps", -999.0) for c in leaderboard]) if leaderboard else winner_pnl
        results.append({
            "cycle": c_num,
            "winner": winner,
            "winner_pnl": winner_pnl,
            "winner_wr": winner_wr,
            "winner_trades": winner_trades,
            "best_pnl": best_pnl,
            "leaderboard": leaderboard
        })
    except Exception as e:
        print(f"Error reading {d}: {e}")

# sort by winner_pnl descending
print("\n" + "="*80)
print(f"{'Cycle':<8} | {'Winner':<15} | {'Trades':<7} | {'Win Rate':<9} | {'Winner Net (bps)':<16} | {'Best Net (bps)':<15}")
print("="*80)
for r in results:
    print(f"Cycle {r['cycle']:<2} | {r['winner']:<15} | {r['winner_trades']:<7} | {r['winner_wr']:<8.1f}% | {r['winner_pnl']:+16.2f} | {r['best_pnl']:+15.2f}")

# Top 10 cycles by PnL
results_by_pnl = sorted(results, key=lambda x: x["winner_pnl"], reverse=True)
print("\n" + "="*80)
print("TOP 10 PEAK CYCLES BY WINNER NET PNL:")
print("="*80)
for r in results_by_pnl[:10]:
    print(f"Cycle {r['cycle']:<2} | {r['winner']:<15} | Trades: {r['winner_trades']:<2} | WR: {r['winner_wr']:5.1f}% | Net: {r['winner_pnl']:+7.2f} bps")
