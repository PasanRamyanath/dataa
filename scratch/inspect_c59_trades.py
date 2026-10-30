import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

log_path = "scratch/archive_cycle59/champion_challenger_final_results.json"
import json
with open(log_path, "r", encoding="utf-8") as f:
    res = json.load(f)

for comp, trades in res.get("competitor_trade_ledgers", {}).items():
    print(f"--- {comp} trades ({len(trades)}) ---")
    for t in trades:
        print(f"Side: {t.get('side')} | Entry: {t.get('entry_price')} | Exit: {t.get('exit_price')} | Net PnL: {t.get('net_return_bps')} bps | Exit Reason: {t.get('exit_reason')}")
