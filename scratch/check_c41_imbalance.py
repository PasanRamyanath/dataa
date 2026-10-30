import json

with open("scratch/live_hierarchical_session_1h_ledger.jsonl") as f:
    for line in f:
        r = json.loads(line)
        if r["bar_idx"] in (171, 172, 244, 245):
            print(f"Bar {r['bar_idx']:3d} | mid={r['bbo_mid']:.2f} | bid_qty={r.get('bbo_bid_qty', 'N/A')} | ask_qty={r.get('bbo_ask_qty', 'N/A')}")
