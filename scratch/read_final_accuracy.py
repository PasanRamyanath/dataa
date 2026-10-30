import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('scratch/live_session_final_metrics.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

print(json.dumps(d.get('accuracy_summary', {}), indent=2))
