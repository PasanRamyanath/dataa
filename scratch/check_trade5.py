import json

with open("scratch/archive_cycle39/live_hierarchical_session_1h_ledger.jsonl") as f:
    for line in f:
        r = json.loads(line)
        if 1793194050000 <= r["ts_ms"] <= 1793194150000:
            bbo_bid = r["bbo_bid"]
            bbo_ask = r["bbo_ask"]
            print(f"Bar {r['bar_idx']}: ts={r['ts_ms']} bid={bbo_bid} ask={bbo_ask} dir={r['primary_direction']} edge={r['expected_net_edge_bps']}")
