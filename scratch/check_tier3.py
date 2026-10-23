import json, sys, io
import requests
from requests.auth import HTTPBasicAuth

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

with open('feature engineering scripts/kaggle.json') as f:
    cfg = json.load(f)

username = cfg['username']
key = cfg['key']
slug = 'btcusdt-v4-ensemble-train'

url = f'https://www.kaggle.com/api/v1/kernels/logs/stream/{username}/{slug}'
auth = HTTPBasicAuth(username, key)
headers = {'Accept': 'text/event-stream, */*'}

events = []
try:
    resp = requests.get(url, headers=headers, auth=auth, stream=True, timeout=(5, 5))
    for line in resp.iter_lines(decode_unicode=True):
        if line and line.startswith('data:'):
            payload = line[5:].strip()
            if payload == 'END_OF_LOG':
                break
            try:
                data = json.loads(payload).get('data', '')
            except Exception:
                data = payload
            events.append(data)
except Exception as e:
    pass

print(f"Total events: {len(events)}")
print("=== LAST 40 EVENTS ===")
for ev in events[-40:]:
    print(ev.rstrip())
