import json
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

trades_file = "scratch/champion_challenger_trade_log.json"
metrics_file = "scratch/champion_challenger_mid_metrics.json"

print("--- CYCLE 59 TELEMETRY CHECK ---")
if os.path.exists(trades_file):
    try:
        with open(trades_file, "r") as f:
            trades = json.load(f)
        print(f"Total trades recorded in log: {len(trades)}")
        for i, t in enumerate(trades):
            print(f"Trade #{i+1}: Competitor: {t.get('competitor') or t.get('name')} | Side: {t.get('side')} | Entry: {t.get('entry_price')} | Exit: {t.get('exit_price')} | PnL: {t.get('net_pnl_bps', t.get('pnl_bps'))} bps | Exit Reason: {t.get('exit_reason')}")
    except Exception as e:
        print(f"Error reading trade log: {e}")
else:
    print("Trade log does not exist yet.")

if os.path.exists(metrics_file):
    try:
        with open(metrics_file, "r") as f:
            metrics = json.load(f)
        print("\nCompetitors Snapshot:")
        for name, data in metrics.items():
            print(f"- {name}: Trades={data.get('trades')}, WinRate={data.get('win_rate_pct')}%, NetPnL={data.get('net_pnl_bps')} bps")
    except Exception as e:
        print(f"Error reading metrics: {e}")
