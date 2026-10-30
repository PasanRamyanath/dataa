import json

bars = [json.loads(l) for l in open('scratch/live_hierarchical_session_1h_ledger.jsonl')]
for i, b in enumerate(bars):
    mid = b['bbo_mid']
    if abs(mid - 80105.90) < 5.0:
        d_mac = b['macro_probs'][2] - b['macro_probs'][0]
        d_mes = b['meso_probs'][2] - b['meso_probs'][0]
        d_tac = b['tactical_probs'][2] - b['tactical_probs'][0]
        d_mic = b['micro_probs'][2] - b['micro_probs'][0]
        print(f"Bar {i:03d}: mid={mid:.2f}, mac={d_mac:+.4f}, mes={d_mes:+.4f}, tac={d_tac:+.4f}, mic={d_mic:+.4f}")
