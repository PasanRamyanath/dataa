import os
import json
import urllib.request
import base64

with open('feature engineering scripts/kaggle_3.json') as f:
    c = json.load(f)

url = 'https://www.kaggle.com/api/v1/kernels/cancel'
req = urllib.request.Request(
    url,
    data=json.dumps({'kernelSlug': 'malaysiatrip/test-gpu-check-malaysiatrip'}).encode(),
    headers={
        'Content-Type': 'application/json',
        'Authorization': 'Basic ' + base64.b64encode(f"{c['username']}:{c['key']}".encode()).decode()
    }
)
try:
    with urllib.request.urlopen(req, timeout=10) as r:
        print('Cancelled:', r.read().decode())
except Exception as e:
    print('Cancel error:', e)
