import json

with open("scratch/archive_cycle11/live_hierarchical_session_1h_ledger.jsonl", "r", encoding="utf-8") as f:
    bars = [json.loads(line) for line in f]

# Trade 1 exited at bar 185 (ts_ms = 1793080170000)
# Trade 2 entered at bar 221 (ts_ms = 1793080530000)
exit_ts = 1793080170000
entry_ts = 1793080530000
elapsed = (entry_ts - exit_ts) / 1000.0

print(f"Elapsed seconds between Trade 1 exit and Trade 2 entry: {elapsed:.1f} seconds ({elapsed/60.0:.1f} minutes)")
