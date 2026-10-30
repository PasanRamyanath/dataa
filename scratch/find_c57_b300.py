import json

with open('scratch/archive_cycle57/live_hierarchical_session_1h_ledger.jsonl') as f:
    for line in f:
        d = json.loads(line)
        if abs(d.get('bbo_bid', 0) - 81728.5) < 1.0:
            print("Bar #", d.get('bar_idx'), "Bid=", d.get('bbo_bid'), "Mid=", d.get('bbo_mid'), "Tact=", d.get('tactical_probs'), "Meso=", d.get('meso_probs'), "Dir=", d.get('primary_direction'), "Edge=", d.get('expected_net_edge_bps'))
