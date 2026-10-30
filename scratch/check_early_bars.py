import json

bars = [json.loads(l) for l in open('scratch/live_hierarchical_session_1h_ledger.jsonl')]
for i in range(15, 30):
    b = bars[i]
    d_mac = b['macro_probs'][2] - b['macro_probs'][0]
    d_mes = b['meso_probs'][2] - b['meso_probs'][0]
    mid = b['bbo_mid']
    print(f"Bar {i:02d}: mid={mid:.2f}, mac={d_mac:+.3f}, mes={d_mes:+.3f}, prod={(d_mac*d_mes):+.4f}, conflict={(d_mac*d_mes) < -0.05}")
