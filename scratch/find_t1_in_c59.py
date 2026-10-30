import json

with open('scratch/archive_cycle59/live_hierarchical_session_1h_ledger.jsonl') as f:
    for line in f:
        d = json.loads(line)
        # Entry of trade 1 for Challenger 3 was at price 81010.9
        mid = d.get('bbo_mid', 0)
        if 80980 <= mid <= 81030:
            print(f"Bar #{d.get('bar_idx')}: Mid={mid}, Bid={d.get('bbo_bid')}, Ask={d.get('bbo_ask')}, Tact={d.get('tactical_probs')}, Meso={d.get('meso_probs')}, Micro={d.get('micro_probs')}, Dir={d.get('primary_direction')}, Edge={d.get('expected_net_edge_bps')}")
