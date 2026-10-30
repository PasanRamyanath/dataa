import json

fpath = "scratch/archive_cycle48/live_hierarchical_session_1h_ledger.jsonl"
with open(fpath, 'r', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        if d['bar_idx'] == 50:
            print("Bar 50 full record:")
            print(json.dumps(d, indent=2))
            break
