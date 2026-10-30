import json

with open("scratch/archive_cycle40/live_hierarchical_session_1h_ledger.jsonl") as f:
    for line in f:
        r = json.loads(line)
        if r["ts_ms"] in (1793196220000, 1793196630000):
            print(f"Bar {r['bar_idx']:3d} | mid={r['bbo_mid']:.2f} | dir={r['primary_direction']} | edge={r['expected_net_edge_bps']:+.2f} | tact={r['tactical_probs']} | meso={r['meso_probs']}")
