import json

with open("scratch/live_hierarchical_session_1h_ledger.jsonl") as f:
    for line in f:
        b = json.loads(line)
        if 60 <= b["bar_idx"] <= 75:
            d_tact = b["tactical_probs"][2] - b["tactical_probs"][0]
            d_meso = b["meso_probs"][2] - b["meso_probs"][0]
            d_micro = b["micro_probs"][2] - b["micro_probs"][0]
            d_macro = b["macro_probs"][2] - b["macro_probs"][0]
            print(f"Bar {b['bar_idx']:03d}: mid={b['bbo_mid']:.2f} dir={b['primary_direction']} edge={b['expected_net_edge_bps']:.2f} pos={bool(b['active_position'])} d_tact={d_tact:+.3f} d_micro={d_micro:+.3f} d_meso={d_meso:+.3f} d_macro={d_macro:+.3f}")
