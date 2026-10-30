import json

with open("scratch/live_hierarchical_session_1h_ledger.jsonl") as f:
    for line in f:
        r = json.loads(line)
        if r["ts_ms"] in (1793200370000, 1793201100000):
            print(f"Bar {r['bar_idx']}: ts={r['ts_ms']} dir={r['primary_direction']} edge={r['expected_net_edge_bps']} bbo_mid={r['bbo_mid']}")
