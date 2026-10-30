import json

with open('scratch/live_hierarchical_session_1h_ledger.jsonl') as f:
    lines = f.readlines()
print("Total lines:", len(lines))
first = json.loads(lines[0])
last = json.loads(lines[-1])
print("First bar:", first.get("bar_idx"), "dt:", first.get("datetime"))
print("Last bar:", last.get("bar_idx"), "dt:", last.get("datetime"))
