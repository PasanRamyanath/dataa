import json

bars_with_trades = []
with open('scratch/live_hierarchical_session_1h_ledger.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        if d.get('closed_trades_count', 0) > 0:
            bars_with_trades.append(d)

print(f"Total bars with closed trades: {len(bars_with_trades)}")
# Look at the transitions where closed_trades_count changes
last_cnt = 0
for b in bars_with_trades:
    cnt = b['closed_trades_count']
    if cnt > last_cnt:
        last_cnt = cnt
        print(f"Bar {b['bar_idx']:3d} | Mid: {b['bbo_mid']} | Closed: {cnt} | Realized PnL: {b['cumulative_realized_pnl_bps']:+.2f} bps")
