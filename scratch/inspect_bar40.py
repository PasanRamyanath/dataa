import json

with open("scratch/live_hierarchical_session_1h_ledger.jsonl") as f:
    bars = [json.loads(line) for line in f if line.strip()]

for i in range(35, 43):
    b = bars[i]
    print(f"Bar {b['bar_idx']:2d} | mid={b['bbo_mid']:.2f} | p_dir={b['primary_direction']} | edge={b.get('expected_net_edge_bps'):.2f} | meso={b['meso_probs']} | tact={b['tactical_probs']} | micro={b['micro_probs']}")
