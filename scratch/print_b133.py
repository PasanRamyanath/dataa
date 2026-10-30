import json

with open('scratch/archive_cycle35/live_hierarchical_session_1h_ledger.jsonl') as f:
    for line in f:
        d = json.loads(line)
        if d.get('bar_idx') == 133:
            print("Bar 133:")
            print("  Tact:", d.get("tactical_probs"))
            print("  Meso:", d.get("meso_probs"))
            print("  Micro:", d.get("micro_probs"))
            print("  Dir:", d.get("primary_direction"))
            print("  Edge:", d.get("expected_net_edge_bps"))
