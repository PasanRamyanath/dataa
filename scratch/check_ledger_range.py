import json

with open('scratch/archive_cycle59/live_hierarchical_session_ledger.jsonl') as f:
    lines = f.readlines()
print(f"Total lines: {len(lines)}")
first = json.loads(lines[0])
last = json.loads(lines[-1])
print("First bar:", first.get("bar_idx"), "ts:", first.get("ts_ms"), "dt:", first.get("datetime"))
print("Last bar:", last.get("bar_idx"), "ts:", last.get("ts_ms"), "dt:", last.get("datetime"))
