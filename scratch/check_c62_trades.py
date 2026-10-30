import sys
import json

sys.stdout.reconfigure(encoding='utf-8')

with open('scratch/archive_cycle62/champion_challenger_final_results.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

for name in ['Champion', 'Challenger_1']:
    print(f"=== {name} ===")
    trades = d['competitor_trade_ledgers'][name]
    for i, t in enumerate(trades, 1):
        print(f"Trade {i}: {t['side']} @ {t['entry_price']} -> {t['exit_price']} | Gross: {t['gross_return_bps']:+.2f}b | Fee: {t['total_fee_bps']}b | Net: {t['net_return_bps']:+.2f}b | Reason: {t['exit_reason']}")
