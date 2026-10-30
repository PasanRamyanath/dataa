import json

bars = [json.loads(l) for l in open('scratch/live_hierarchical_session_1h_ledger.jsonl')]
for i, b in enumerate(bars):
    mid = b['bbo_mid']
    if mid > 80150:
        print(f"Bar {i:03d} (ts={b['ts_ms']}): mid={mid:.2f}, bid={b['bbo_bid']:.2f}, ask={b['bbo_ask']:.2f}")
        print(f"  macro: {b['macro_probs']}, meso: {b['meso_probs']}, tac: {b['tactical_probs']}, mic: {b['micro_probs']}")
        d_mes = b['meso_probs'][2] - b['meso_probs'][0]
        d_tac = b['tactical_probs'][2] - b['tactical_probs'][0]
        d_mic = b['micro_probs'][2] - b['micro_probs'][0]
        print(f"  deltas: meso={d_mes:+.4f}, tac={d_tac:+.4f}, mic={d_mic:+.4f}")
