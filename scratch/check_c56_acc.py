import json

with open('scratch/live_session_1h_final_metrics.json') as f:
    data = json.load(f)

print('Session Duration Min:', data.get('session_duration_minutes'))
print('Bars evaluated:', data.get('bars_evaluated'))
print('Total Trades:', data.get('total_trades'))
print('Winning Trades:', data.get('winning_trades'))
print('Total Net PnL bps:', data.get('total_net_pnl_bps'))
print('Accuracy Summary:')
for h, acc in data.get('accuracy_summary', {}).items():
    print(f"  {h}: Overall={acc.get('overall_accuracy_pct')}% | Active Precision={acc.get('active_precision_pct')}% | Active Count={acc.get('active_signals_count')}")
