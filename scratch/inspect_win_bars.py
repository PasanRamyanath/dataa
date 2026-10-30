import json

with open('scratch/archive_cycle61/live_hierarchical_session_1h_ledger.jsonl', 'r') as f:
    for line in f:
        d = json.loads(line)
        idx = d.get('bar_idx', 0)
        if idx in [17, 18, 19, 173, 174, 175, 225, 226, 227]:
            tact = d.get('tactical_probs', [])
            micro = d.get('micro_probs', [])
            meso = d.get('meso_probs', [])
            edge = d.get('expected_net_edge_bps', 0)
            d_tact = tact[2] - tact[0] if len(tact) == 3 else 0
            d_micro = micro[2] - micro[0] if len(micro) == 3 else 0
            d_meso = meso[2] - meso[0] if len(meso) == 3 else 0
            print(f"Bar {idx:03d} | Mid: {d['bbo_mid']:.1f} | d_tact: {d_tact:+.3f} | d_micro: {d_micro:+.3f} | d_meso: {d_meso:+.3f} | edge: {edge:.1f}b | mode: {d.get('execution_mode')}")
