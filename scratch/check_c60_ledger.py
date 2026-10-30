import json

with open('scratch/archive_cycle60/live_hierarchical_session_1h_ledger.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        d = json.loads(line)
        print(f"Bar {i}: mid={d.get('bbo_mid')} dir={d.get('primary_direction')} mode={d.get('execution_mode')} p_down={d.get('meso_probs', [0])[0]:.2f} p_up={d.get('meso_probs', [0,0,0])[2]:.2f}")
