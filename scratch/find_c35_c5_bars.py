import json

ts1 = 1793175670000
ts2 = 1793176430000

with open("scratch/archive_cycle35/live_hierarchical_session_1h_ledger.jsonl", 'r', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        if d['ts_ms'] in [ts1, ts2]:
            print(f"Bar {d['bar_idx']:03d} | TS: {d['ts_ms']} | Price: {d['bbo_mid']} | Tact: {d['tactical_probs']} | Meso: {d['meso_probs']}")
