import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('scratch/archive_cycle62/champion_challenger_final_results.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

trades = d['competitor_trade_ledgers']['Champion']
for i, t in enumerate(trades, 1):
    print(f"Trade {i}: entry_ts={t['entry_ts_ms']} exit_ts={t['exit_ts_ms']} entry_p={t['entry_price']} exit_p={t['exit_price']}")
    if i > 1:
        prev = trades[i-2]
        gap_sec = (t['entry_ts_ms'] - prev['exit_ts_ms']) / 1000.0
        print(f"   Gap from prev exit: {gap_sec:.1f}s | prev exit price: {prev['exit_price']} | entry price: {t['entry_price']}")
