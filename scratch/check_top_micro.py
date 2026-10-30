import json

bars = [json.loads(l) for l in open('scratch/live_hierarchical_session_1h_ledger.jsonl')]
for i in range(230, 246):
    b = bars[i]
    print(f"Bar {i}: mid={b['bbo_mid']:.2f}, bid={b['bbo_bid']:.2f}, ask={b['bbo_ask']:.2f}")
    print(f"  micro: {b['micro_probs']}")
