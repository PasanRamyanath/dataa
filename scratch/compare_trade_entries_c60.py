import json

with open('scratch/archive_cycle60/live_hierarchical_session_1h_ledger.jsonl', 'r') as f:
    for line in f:
        d = json.loads(line)
        if d['bar_idx'] in [107, 108, 117, 118, 155, 156, 209, 210, 239, 240, 241]:
            print(f"Bar {d['bar_idx']}: mid={d['bbo_mid']} | macro={d['macro_probs']} | meso={d['meso_probs']} | tact={d['tactical_probs']} | micro={d['micro_probs']} | edge={d['expected_net_edge_bps']}")
