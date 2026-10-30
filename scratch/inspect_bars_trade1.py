import json

with open('scratch/live_hierarchical_session_1h_ledger.jsonl') as f:
    lines = [json.loads(l) for l in f]

for i in range(137, 150):
    l = lines[i]
    print(f"Bar {i} (ts={l['ts_ms']}): mid={l['bbo_mid']} | fsm={l['fsm_state']} | mode={l['execution_mode']} | dir={l['primary_direction']} | pos={l['active_position'] is not None}")
