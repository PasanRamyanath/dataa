import json

bars = [json.loads(l) for l in open('scratch/live_hierarchical_session_1h_ledger.jsonl')]
for i in range(235, 280):
    b = bars[i]
    mid = b['bbo_mid']
    d_mes = b['meso_probs'][2] - b['meso_probs'][0]
    d_tac = b['tactical_probs'][2] - b['tactical_probs'][0]
    d_mic = b['micro_probs'][2] - b['micro_probs'][0]
    print(f"Bar {i:03d} (ts={b['ts_ms']}): mid={mid:.2f} | mes={d_mes:+.3f}, tac={d_tac:+.3f}, mic={d_mic:+.3f}")
