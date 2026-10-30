import json

with open('scratch/live_hierarchical_session_1h_ledger.jsonl', 'r') as f:
    for line in f:
        d = json.loads(line)
        pos = d.get('active_position')
        if pos and pos.get('side') == 'SELL':
            print("Bar", d['bar_idx'], "mid:", d['bbo_mid'], "macro:", d['macro_probs'], "meso:", d['meso_probs'], "tact:", d['tactical_probs'])
