import json

bars = [json.loads(l) for l in open('scratch/live_hierarchical_session_1h_ledger.jsonl')]

# Look for transitions in closed_trades_count
trade_bars = []
last_cnt = 0
for i, l in enumerate(bars):
    cnt = l.get('closed_trades_count', 0)
    if cnt > last_cnt:
        trade_bars.append((cnt, i, l['ts_ms'], l['bbo_mid']))
        last_cnt = cnt

print(f"Closed trades count: {last_cnt}")
for t in trade_bars:
    print(f"Trade {t[0]} closed at Bar {t[1]} (ts={t[2]}): mid={t[3]}")

# Look at position entries
for i in range(1, len(bars)):
    cur_pos = bars[i].get('active_position')
    prev_pos = bars[i-1].get('active_position')
    if cur_pos is not None and prev_pos is None:
        print(f"Entry at Bar {i}: side={cur_pos['side']}, entry_price={cur_pos['entry_price']}, mid={bars[i]['bbo_mid']}")
    if cur_pos is None and prev_pos is not None:
        gross = (bars[i]['bbo_mid'] - prev_pos['entry_price']) / prev_pos['entry_price'] * 1e4
        if prev_pos['side'] == 'SELL': gross = -gross
        print(f"Exit at Bar {i}: side={prev_pos['side']}, entry={prev_pos['entry_price']}, exit_mid={bars[i]['bbo_mid']}, gross={gross:.2f} bps, dur={prev_pos.get('holding_sec')}s")
