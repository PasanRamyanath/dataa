import json

with open('scratch/live_hierarchical_session_1h_ledger.jsonl') as f:
    lines = [json.loads(l) for l in f]

for i, l in enumerate(lines):
    if l.get('closed_trades_count') == 2:
        print(f"Trade 2 closed at Bar {i} (ts={l['ts_ms']}): mid={l['bbo_mid']}")
        break

# Look at bars around 180 to 276 for active position
for i in range(180, len(lines)):
    l = lines[i]
    if l.get('active_position') is not None:
        ap = l['active_position']
        print(f"Bar {i} (ts={l['ts_ms']}): mid={l['bbo_mid']}, side={ap['side']}, entry={ap['entry_price']}, mfe={ap['peak_favorable_excursion_bps']}, armed={ap['trailing_maker_armed']}")
