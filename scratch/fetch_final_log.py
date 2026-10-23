import json
import requests
from requests.auth import HTTPBasicAuth

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
    resp = requests.get(url, headers=headers, auth=auth, stream=True, timeout=(8, 15))
    for line in resp.iter_lines(decode_unicode=True):
        if line and line.startswith('data:'):
            p = line[5:].strip()
            if p == 'END_OF_LOG':
                print('[Got END_OF_LOG]')
                break
            try:
                d = json.loads(p).get('data', '')
            except Exception:
                d = p
            events.append(d)
        if len(events) > 8000:
            events = events[-2500:]
except Exception as e:
    print('Exception:', e)

print(f'Total events retrieved: {len(events)}')
with open('scratch/v9_final_tail.txt', 'w', encoding='utf-8') as f:
    for ev in events[-60:]:
        f.write(ev)
        if not ev.endswith('\n'):
            f.write('\n')
print('Wrote last 60 events to scratch/v9_final_tail.txt')
