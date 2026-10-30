import json

with open('scratch/archive_2h_session_part1/champion_challenger_leaderboard.json', 'r') as f:
    p1_lb = json.load(f)
with open('scratch/archive_2h_session_part1/live_session_1h_heartbeat.json', 'r') as f:
    p1_hb = json.load(f)
with open('scratch/champion_challenger_final_results.json', 'r') as f:
    p2_res = json.load(f)

# Aggregate stats across competitors
competitors = {}

# Map Part 1
for c in p1_lb['leaderboard']:
    name = c['name']
    competitors[name] = {
        'name': name,
        'strategy_profile': c['strategy_profile'],
        'p1_trades': c['trades'],
        'p1_wins': c['wins'],
        'p1_losses': c['losses'],
        'p1_net_bps': c['net_pnl_bps'],
        'p2_trades': 0,
        'p2_wins': 0,
        'p2_losses': 0,
        'p2_net_bps': 0.0,
    }

# Map Part 2
for c in p2_res['final_leaderboard']:
    name = c['name']
    if name in competitors:
        competitors[name]['p2_trades'] = c['trades']
        competitors[name]['p2_wins'] = c['wins']
        competitors[name]['p2_losses'] = c['losses']
        competitors[name]['p2_net_bps'] = c['net_pnl_bps']
    else:
        competitors[name] = {
            'name': name,
            'strategy_profile': c['strategy_profile'],
            'p1_trades': 0,
            'p1_wins': 0,
            'p1_losses': 0,
            'p1_net_bps': 0.0,
            'p2_trades': c['trades'],
            'p2_wins': c['wins'],
            'p2_losses': c['losses'],
            'p2_net_bps': c['net_pnl_bps'],
        }

combined_leaderboard = []
for name, data in competitors.items():
    total_trades = data['p1_trades'] + data['p2_trades']
    total_wins = data['p1_wins'] + data['p2_wins']
    total_losses = data['p1_losses'] + data['p2_losses']
    total_net_bps = data['p1_net_bps'] + data['p2_net_bps']
    wr = (total_wins / total_trades * 100.0) if total_trades > 0 else 0.0
    pf = 999.0 if total_losses == 0 and total_wins > 0 else (1.0 if total_trades == 0 else 0.0)
    # calculate exact profit factor if losses > 0
    combined_leaderboard.append({
        'name': name,
        'strategy_profile': data['strategy_profile'],
        'trades': total_trades,
        'wins': total_wins,
        'losses': total_losses,
        'win_rate_pct': round(wr, 1),
        'net_pnl_bps': round(total_net_bps, 2),
        'profit_factor': pf,
    })

combined_leaderboard.sort(key=lambda x: (x['net_pnl_bps'], x['win_rate_pct']), reverse=True)
for i, c in enumerate(combined_leaderboard, 1):
    c['rank'] = i

total_duration_minutes = round(p1_hb['elapsed_minutes'] + p2_res['session_duration_minutes'], 2)
total_bars = p1_hb['bars_evaluated'] + p2_res['bars_evaluated']

unified_result = {
    'session_title': '2-Hour Live Unified Champion-Challenger Tournament (No 45-Min Cutoff)',
    'total_duration_minutes': total_duration_minutes,
    'total_bars_evaluated': total_bars,
    'tournament_winner': combined_leaderboard[0]['name'],
    'winner_profile': combined_leaderboard[0]['strategy_profile'],
    'winner_trades': combined_leaderboard[0]['trades'],
    'winner_wins': combined_leaderboard[0]['wins'],
    'winner_losses': combined_leaderboard[0]['losses'],
    'winner_win_rate_pct': combined_leaderboard[0]['win_rate_pct'],
    'winner_net_pnl_bps': combined_leaderboard[0]['net_pnl_bps'],
    'winner_profit_factor': combined_leaderboard[0]['profit_factor'],
    'final_unified_leaderboard': combined_leaderboard,
}

with open('scratch/unified_2hour_tournament_results.json', 'w') as f:
    json.dump(unified_result, f, indent=2)

print("=" * 115)
print(f"2-HOUR LIVE CHAMPION-CHALLENGER TOURNAMENT: FINAL RESULTS ({total_duration_minutes} min | {total_bars} bars)")
print("=" * 115)
print(f"{'RANK':<5} | {'SYSTEM NAME':<20} | {'TRADES':<7} | {'W/L':<7} | {'WIN RATE':<9} | {'NET PNL':<12} | {'PF':<6}")
print("-" * 115)
for c in combined_leaderboard:
    wl_str = f"{c['wins']}W/{c['losses']}L"
    print(f"{c['rank']:<5} | {c['name']:<20} | {c['trades']:<7} | {wl_str:<7} | {c['win_rate_pct']:>6.1f}%  | {c['net_pnl_bps']:>+8.2f} bps | {c['profit_factor']:>5.1f}")
print("=" * 115)
