import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('scratch/live_session_1h_heartbeat.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

print(json.dumps(d, indent=2))
