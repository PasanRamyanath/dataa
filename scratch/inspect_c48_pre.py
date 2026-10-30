import json

fpath = "scratch/archive_cycle48/live_hierarchical_session_1h_ledger.jsonl"
with open(fpath, 'r', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        idx = d['bar_idx']
        if idx in [10, 50, 100, 150, 200, 250, 270, 274]:
            t_p = d['tactical_probs']
            m_p = d['meso_probs']
            mac_p = d['macro_probs']
            print(f"Bar {idx:03d} | Mid: {d['bbo_mid']} | Macro: {mac_p} | Meso: {m_p} | Tact: {t_p} | Edge: {d['expected_net_edge_bps']}")
