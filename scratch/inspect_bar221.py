import json

with open("scratch/archive_cycle11/live_hierarchical_session_1h_ledger.jsonl", "r", encoding="utf-8") as f:
    bars = [json.loads(line) for line in f]

for idx in [220, 221, 222]:
    b = bars[idx-1]
    print(f"--- BAR {idx} ---")
    for k, v in b.items():
        print(f"  {k}: {v}")
