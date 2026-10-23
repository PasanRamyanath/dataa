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
    resp = requests.get(url, headers=headers, auth=auth, stream=True, timeout=(5, 6))
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
        if len(events) > 8000:
            events = events[-3000:]
except requests.exceptions.Timeout:
    pass
except Exception as e:
    print('Error:', e)

print(f"Total events: {len(events)}")
print("=== LAST 30 EVENTS ===")
for ev in events[-30:]:
    print(ev.rstrip())
