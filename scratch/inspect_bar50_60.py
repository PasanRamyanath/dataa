import json

with open("scratch/live_hierarchical_session_1h_ledger.jsonl") as f:
    bars = [json.loads(line) for line in f if line.strip()]

for i in range(50, 60):
    b = bars[i]
    print(f"Bar {b['bar_idx']}: bid={b['bbo_bid']} ask={b['bbo_ask']} meso={b['meso_probs']} tact={b['tactical_probs']} micro={b['micro_probs']}")
