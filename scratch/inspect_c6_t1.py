import json

bars = [json.loads(l) for l in open('scratch/live_hierarchical_session_1h_ledger.jsonl')]

for i, l in enumerate(bars):
    if l.get('closed_trades_count', 0) > 0:
        print(f"Trade closed at Bar {i} (ts={l['ts_ms']}): mid={l['bbo_mid']}")
        break

for i in range(len(bars)):
    if bars[i].get('active_position') is not None:
        ap = bars[i]['active_position']
        print(f"Active at Bar {i}: side={ap['side']}, entry={ap['entry_price']}, mfe={ap['peak_favorable_excursion_bps']}")
        break
