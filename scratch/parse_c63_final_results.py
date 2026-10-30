import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('scratch/champion_challenger_final_results.json', 'r', encoding='utf-8') as f:
    res = json.load(f)

print(f"Session Duration: {res.get('session_duration_minutes')} min | Bars Evaluated: {res.get('bars_evaluated')}")
print(f"Tournament Winner: {res.get('tournament_winner')}")
print(f"Winner Profile: {res.get('winner_profile')}")
print(f"Winner Net PnL: {res.get('winner_net_pnl_bps'):+.2f} bps | WR: {res.get('winner_win_rate_pct')}% | Trades: {res.get('winner_trades')}")

print("\n" + "=" * 105)
print("FINAL LEADERBOARD:")
print(f"{'Rank':4s} | {'Competitor':14s} | {'Trades':6s} | {'Wins':4s} | {'Losses':6s} | {'Win Rate':8s} | {'Net PnL':10s} | {'PF':5s} | Strategy Profile")
print("=" * 105)
for entry in res.get('final_leaderboard', []):
    rank = entry.get('rank')
    name = entry.get('name')
    trades = entry.get('trades')
    wins = entry.get('wins')
    losses = entry.get('losses')
    wr = entry.get('win_rate_pct')
    net = entry.get('net_pnl_bps')
    pf = entry.get('profit_factor')
    prof = entry.get('strategy_profile')
    print(f"#{rank:<3d} | {name:14s} | {trades:6d} | {wins:4d} | {losses:6d} | {wr:6.1f}% | {net:+8.2f}b | {pf:5.2f} | {prof}")

print("\n" + "=" * 105)
print("COMPETITOR TRADE LEDGERS:")
ledgers = res.get('competitor_trade_ledgers', {})
for name, trades in ledgers.items():
    print(f"\n--- {name} (Total Trades: {len(trades)}) ---")
    for i, t in enumerate(trades, 1):
        print(f"  Trade {i}: {t['side']:4s} @ {t['entry_price']:.2f} -> {t['exit_price']:.2f} | Gross: {t['gross_return_bps']:+6.2f}b | Fee: {t['total_fee_bps']:3.1f}b | Net: {t['net_return_bps']:+6.2f}b | MFE: {t['peak_mfe_bps']:+5.2f}b | MAE: {t['max_mae_bps']:+5.2f}b | Dur: {t['holding_duration_sec']:.0f}s | Reason: {t['exit_reason']}")
