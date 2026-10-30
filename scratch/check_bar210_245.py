import json

with open("scratch/live_hierarchical_session_1h_ledger.jsonl") as f:
    for line in f:
        r = json.loads(line)
        if 210 <= r["bar_idx"] <= 245 and r["bar_idx"] % 5 == 0:
            print(f"Bar {r['bar_idx']:3d} | mid={r['bbo_mid']:.2f} | dir={r['primary_direction']} | edge={r['expected_net_edge_bps']:+.2f}")
