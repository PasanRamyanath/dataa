import json

fpath = "scratch/archive_cycle48/live_hierarchical_session_1h_ledger.jsonl"
with open(fpath, 'r', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        if 135 <= d['bar_idx'] <= 140:
            print(f"Bar {d['bar_idx']:03d} | Price: {d['bbo_mid']} | Dir: {d['primary_direction']} | Edge: {d['expected_net_edge_bps']} | Mode: {d['execution_mode']}")
