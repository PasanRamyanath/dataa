import json

with open("scratch/archive_cycle35/live_hierarchical_session_1h_ledger.jsonl", "r") as f:
    lines = [json.loads(l) for l in f]

# Find bars around Trade 1 entry (81123.2) and Trade 2 entry (81192.3)
print("=== CYCLE 35: TRADE 1 ENTRY (LOSS, MFE=0) ===")
for r in lines:
    if abs(r.get("bbo_bid", 0) - 81123.2) < 2.0:
        print(f"Bar #{r['bar_idx']}: Mid={r['bbo_mid']}, Tact={r['tactical_probs']}, Meso={r['meso_probs']}, Micro={r['micro_probs']}, Dir={r['primary_direction']}, Edge={r['expected_net_edge_bps']}")

print("\n=== CYCLE 35: TRADE 2 ENTRY (WIN, MFE=+21.6b) ===")
for r in lines:
    if abs(r.get("bbo_bid", 0) - 81192.3) < 2.0:
        print(f"Bar #{r['bar_idx']}: Mid={r['bbo_mid']}, Tact={r['tactical_probs']}, Meso={r['meso_probs']}, Micro={r['micro_probs']}, Dir={r['primary_direction']}, Edge={r['expected_net_edge_bps']}")
