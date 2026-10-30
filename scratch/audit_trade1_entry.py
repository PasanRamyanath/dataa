import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Let's inspect the ledger from Cycle 59 to see the exact bar where Trade 1 was entered
import json
ledger_file = "scratch/archive_cycle59/live_hierarchical_session_ledger.jsonl"
with open(ledger_file, "r") as f:
    for line in f:
        rec = json.loads(line)
        if rec.get("bar_index", 0) <= 25:
            # check if there's any trade event or check price
            pass

print("Ledger loaded.")
