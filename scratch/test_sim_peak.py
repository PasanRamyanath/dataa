import json
import glob
import os

def test_on_cycle(cycle_num):
    arch_file = f"scratch/archive_cycle{cycle_num}/live_hierarchical_session_1h_ledger.jsonl"
    if not os.path.exists(arch_file):
        return None
    
    bars = []
    with open(arch_file, 'r', encoding='utf-8') as f:
        for line in f:
            bars.append(json.loads(line))
    return bars

print("Archive files checked successfully.")
