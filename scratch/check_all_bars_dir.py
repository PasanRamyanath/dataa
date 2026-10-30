import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('scratch/live_hierarchical_session_1h_ledger.jsonl', encoding='utf-8') as f:
    lines = [json.loads(line) for line in f if line.strip()]

dir_counts = {0: 0, 1: 0, 2: 0}
for d in lines:
    dir_counts[d.get('primary_direction')] = dir_counts.get(d.get('primary_direction'), 0) + 1

print(f"Total bars: {len(lines)}")
print("Direction counts (0=SHORT, 1=FLAT, 2=LONG):", dir_counts)
