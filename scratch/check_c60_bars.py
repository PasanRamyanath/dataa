import json

with open('scratch/archive_cycle60/live_hierarchical_session_1h_ledger.jsonl', 'r') as f:
    for line in f:
        d = json.loads(line)
        if d['bar_idx'] in [107, 108, 114, 115, 154, 155, 208, 209, 239, 240]:
            print(f"Bar {d['bar_idx']}: mode={d.get('execution_mode')}, fsm={d.get('fsm_state')}, dir={d.get('primary_direction')}, edge={d.get('expected_net_edge_bps')}")
