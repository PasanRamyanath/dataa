import json

with open('scratch/archive_cycle35/champion_challenger_final_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for name, trades in data['competitor_trade_ledgers'].items():
    print(f"--- {name} ({len(trades)} trades) ---")
    for t in trades:
        print(f"  {t['side']} @ {t['entry_price']} -> {t['exit_price']} | Net: {t['net_return_bps']:+.2f} bps | Reason: {t['exit_reason']}")
