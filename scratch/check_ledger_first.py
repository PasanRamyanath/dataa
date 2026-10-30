import json
with open('scratch/live_hierarchical_session_1h_ledger.jsonl') as f:
    d = json.loads(f.readline())
print("bbo_bid:", d.get('bbo_bid'))
print("bbo_ask:", d.get('bbo_ask'))
print("bbo_mid:", d.get('bbo_mid'))
