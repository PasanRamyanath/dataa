import json
import os

winners = [(48, 63), (48, 273), (35, 152), (35, 194), (35, 209), (43, 196), (43, 248)]
losers = [(35, 12), (35, 66), (43, 222), (43, 343), (48, 158), (48, 337)]

def inspect_bar(cycle_id, bar_idx):
    fpath = f"scratch/archive_cycle{cycle_id}/live_hierarchical_session_1h_ledger.jsonl"
    with open(fpath, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            if d['bar_idx'] == bar_idx:
                return d
    return None

print("=== WINNERS ===")
for c, b in winners:
    d = inspect_bar(c, b)
    if d:
        d_tact = d['tactical_probs'][2] - d['tactical_probs'][0]
        d_meso = d['meso_probs'][2] - d['meso_probs'][0]
        d_mic = d['micro_probs'][2] - d['micro_probs'][0]
        d_mac = d['macro_probs'][2] - d['macro_probs'][0]
        print(f"C{c} Bar {b:03d} | Edge: {d['expected_net_edge_bps']:4.1f} | d_mic: {d_mic:+.3f} | d_tact: {d_tact:+.3f} | d_meso: {d_meso:+.3f} | d_mac: {d_mac:+.3f}")

print("\n=== LOSERS ===")
for c, b in losers:
    d = inspect_bar(c, b)
    if d:
        d_tact = d['tactical_probs'][2] - d['tactical_probs'][0]
        d_meso = d['meso_probs'][2] - d['meso_probs'][0]
        d_mic = d['micro_probs'][2] - d['micro_probs'][0]
        d_mac = d['macro_probs'][2] - d['macro_probs'][0]
        print(f"C{c} Bar {b:03d} | Edge: {d['expected_net_edge_bps']:4.1f} | d_mic: {d_mic:+.3f} | d_tact: {d_tact:+.3f} | d_meso: {d_meso:+.3f} | d_mac: {d_mac:+.3f}")
