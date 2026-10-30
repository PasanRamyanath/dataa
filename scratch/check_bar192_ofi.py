import json

with open("scratch/live_hierarchical_session_1h_ledger.jsonl") as f:
    bars = [json.loads(line) for line in f if line.strip()]

for i in range(188, 196):
    b = bars[i]
    bid_qty = b.get("bbo_bid_qty", 1.0)
    ask_qty = b.get("bbo_ask_qty", 1.0)
    imb = bid_qty / max(ask_qty, 1e-4)
    print(f"Bar {b['bar_idx']:3d} | mid={b['bbo_mid']:.2f} | p_dir={b['primary_direction']} | edge={b.get('expected_net_edge_bps'):.2f} | bid_qty={bid_qty:.3f} | ask_qty={ask_qty:.3f} | OFI_imb={imb:.3f}")
