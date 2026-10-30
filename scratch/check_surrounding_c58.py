import json

with open('scratch/archive_cycle58/live_hierarchical_session_1h_ledger.jsonl', 'r', encoding='utf-8') as f:
    bars = [json.loads(line) for line in f if line.strip()]

target_idx = None
for idx, b in enumerate(bars):
    if 81470 <= b['bbo_mid'] <= 81480:
        target_idx = idx
        break

print(f"Target Bar Index: {target_idx} | Mid: {bars[target_idx]['bbo_mid']}")
for k in range(max(0, target_idx - 6), min(len(bars), target_idx + 10)):
    m = bars[k]['bbo_mid']
    e = bars[k]['expected_net_edge_bps']
    pdir = bars[k]['primary_direction']
    print(f"Bar {k:3d} | Mid: {m:8.2f} | Dir: {pdir} | Edge: {e:5.1f}")
