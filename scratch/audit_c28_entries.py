import json

with open('scratch/live_hierarchical_session_1h_ledger.jsonl', 'r') as f:
    for line in f:
        d = json.loads(line)
        if d['bar_idx'] in (20, 21, 22, 180, 181, 246, 247, 272, 273, 274):
            print(f"Bar {d['bar_idx']:03d} | mid={d['bbo_mid']:.2f} | macro={d['macro_probs']} | meso={d['meso_probs']} | tact={d['tactical_probs']} | dir={d['primary_direction']}")
