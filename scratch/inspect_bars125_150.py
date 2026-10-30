import json
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

with open('scratch/archive_cycle59/live_hierarchical_session_1h_ledger.jsonl') as f:
    for line in f:
        d = json.loads(line)
        b = d.get('bar_idx', 0)
        if 125 <= b <= 150:
            print(f"Bar #{b:03d}: Mid={d['bbo_mid']:.2f}, Tact={d['tactical_probs']}, Meso={d['meso_probs']}, Micro={d['micro_probs']}, Dir={d['primary_direction']}")
