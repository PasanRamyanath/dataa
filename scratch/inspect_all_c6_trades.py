import json

bars = [json.loads(l) for l in open('scratch/live_hierarchical_session_1h_ledger.jsonl')]

for i in range(1, len(bars)):
    cur_cnt = bars[i].get('closed_trades_count', 0)
    prev_cnt = bars[i-1].get('closed_trades_count', 0)
    if cur_cnt > prev_cnt:
        print(f"Trade {cur_cnt} closed at Bar {i} (ts={bars[i]['ts_ms']}): mid={bars[i]['bbo_mid']:.2f}")

# Look at closed trades stored in the engine by inspecting log or positions
