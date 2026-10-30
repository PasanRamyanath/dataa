import json

bars = [json.loads(l) for l in open('scratch/live_hierarchical_session_1h_ledger.jsonl')]

for idx in [61, 103, 182]:
    b = bars[idx]
    d_mac = b['macro_probs'][2] - b['macro_probs'][0]
    d_mes = b['meso_probs'][2] - b['meso_probs'][0]
    d_tac = b['tactical_probs'][2] - b['tactical_probs'][0]
    d_mic = b['micro_probs'][2] - b['micro_probs'][0]
    print(f"Bar {idx} (Exit): mid={b['bbo_mid']:.2f}")
    print(f"  deltas: mac={d_mac:+.4f}, mes={d_mes:+.4f}, tac={d_tac:+.4f}, mic={d_mic:+.4f}")
