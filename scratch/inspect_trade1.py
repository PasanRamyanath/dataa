import json

with open('scratch/live_hierarchical_session_1h_ledger.jsonl') as f:
    lines = [json.loads(l) for l in f]

for i, l in enumerate(lines):
    if l.get('closed_trades_count', 0) > 0:
        print(f"Trade closed at bar {i} (ts={l['ts_ms']}): mid={l['bbo_mid']}, dir={l['primary_direction']}, fsm={l['fsm_state']}")
        break

for i, l in enumerate(lines):
    if l.get('active_position') is not None:
        ap = l['active_position']
        print(f"Bar {i}: active_position={ap}")
        break
