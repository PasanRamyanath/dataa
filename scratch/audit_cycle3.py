import json

with open('scratch/live_hierarchical_session_1h_ledger.jsonl', 'r') as f:
    lines = [json.loads(line) for line in f]

p_start = lines[0]['bbo_mid']
p_end = lines[-1]['bbo_mid']
p_min = min(l['bbo_mid'] for l in lines)
p_max = max(l['bbo_mid'] for l in lines)

print(f"Start Price: {p_start}, End Price: {p_end}, Min: {p_min}, Max: {p_max}")
print(f"Overall Excursion: {((p_end - p_start)/p_start)*10000:.2f} bps, Max Excursion: {((p_max - p_start)/p_start)*10000:.2f} bps, Min Excursion: {((p_min - p_start)/p_start)*10000:.2f} bps")

# Look at macro and meso probs at key bars
for idx in [1, 38, 39, 60, 119, 120, 125, 126, 180, 200, 257, 300, 335, 360]:
    b = next((l for l in lines if l['bar_idx'] == idx), None)
    if b:
        print(f"Bar {idx:3d} | Mid: {b['bbo_mid']:8.2f} | Dir: {b['primary_direction']} | Edge: {b['expected_net_edge_bps']:5.1f} | Macro: {b['macro_probs']} | Meso: {b['meso_probs']} | Tact: {b['tactical_probs']} | Micro: {b['micro_probs']}")
