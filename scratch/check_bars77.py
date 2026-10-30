import json

with open("scratch/archive_cycle36/live_hierarchical_session_1h_ledger.jsonl") as f:
    bars = [json.loads(line) for line in f if line.strip()]

for i in range(77, 85):
    b = bars[i]
    print(f"Bar {i}: bid={b['bbo_bid']}, ask={b['bbo_ask']}, p_dir={b['primary_direction']}, edge={b.get('expected_net_edge_bps')}")
