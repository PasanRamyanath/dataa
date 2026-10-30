import json

with open("scratch/live_hierarchical_session_1h_ledger.jsonl") as f:
    for line in f:
        r = json.loads(line)
        if r["bar_idx"] in (170, 171, 172, 173, 174, 175, 243, 244, 245, 246, 247):
            print(f"Bar {r['bar_idx']:3d} | mid={r['bbo_mid']:.2f} | dir={r['primary_direction']} | edge={r['expected_net_edge_bps']:+.2f} | macro={r['macro_probs']} | meso={r['meso_probs']} | tact={r['tactical_probs']} | micro={r['micro_probs']}")
