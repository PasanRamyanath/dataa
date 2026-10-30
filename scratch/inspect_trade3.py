import json

bars = [json.loads(l) for l in open('scratch/archive_cycle5/live_hierarchical_session_1h_ledger.jsonl')]

for i in range(230, 300):
    l = bars[i]
    if l.get('active_position') is not None and l['active_position']['side'] == 'BUY':
        ap = l['active_position']
        if abs(ap['entry_price'] - 79872.60) < 1.0:
            print(f"Trade 3 active at Bar {i} (ts={l['ts_ms']}): mid={l['bbo_mid']:.2f}, entry={ap['entry_price']}")
            print(f"  macro: {l['macro_probs']}, meso: {l['meso_probs']}, tac: {l['tactical_probs']}")
