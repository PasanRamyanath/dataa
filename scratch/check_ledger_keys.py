import json

with open('scratch/live_hierarchical_session_1h_ledger.jsonl') as f:
    first_line = json.loads(f.readline().strip())
    print("Keys in ledger:", list(first_line.keys()))
