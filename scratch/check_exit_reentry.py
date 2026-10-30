import json

bars = [json.loads(l) for l in open('scratch/live_hierarchical_session_1h_ledger.jsonl')]
for i, b in enumerate(bars):
    mid = b['bbo_mid']
    if abs(mid - 80172.60) < 1.0 or abs(mid - 80177.90) < 1.0:
        print(f"Bar {i:03d}: mid={mid:.2f}, ts={b['ts_ms']}")
