import json

with open('scratch/archive_cycle59/live_hierarchical_session_ledger.jsonl') as f:
    for line in f:
        d = json.loads(line)
        ts = d.get('timestamp_ms', 0)
        if 1793295550000 <= ts <= 1793295750000:
            print(f"Bar {d.get('bar_index')}: Mid={d.get('mid_price')}, Tact_probs={d.get('tactical_probs')}, Meso_probs={d.get('meso_probs')}, Micro_probs={d.get('micro_probs')}")
