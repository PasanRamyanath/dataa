import os
import glob
import json

base_dir = 'scratch'
for arch, bar_idx in [('archive_cycle43', 167), ('archive_cycle52', 49), ('archive_cycle57', 135)]:
    ledgers = glob.glob(os.path.join(base_dir, arch, '*ledger*.jsonl'))
    if not ledgers: continue
    with open(ledgers[0], 'r', encoding='utf-8') as f:
        bars = [json.loads(l) for l in f if l.strip().startswith('{')]
    mids = [b.get('bbo_mid', b.get('mid')) for b in bars]
    
    print(f"\n==================== {arch} Bar {bar_idx} Pre-Entry Window ====================")
    for k in range(max(0, bar_idx - 10), min(len(bars), bar_idx + 5)):
        b = bars[k]
        mid = mids[k]
        macro = b.get('macro_probs', [0.33, 0.33, 0.33])
        meso = b.get('meso_probs', [0.33, 0.33, 0.33])
        tact = b.get('tactical_probs', [0.33, 0.33, 0.33])
        p_dir = b.get('primary_direction', 1)
        edge = b.get('expected_net_edge_bps', 0.0)
        d_macro = macro[2] - macro[0]
        d_meso = meso[2] - meso[0]
        d_tact = tact[2] - tact[0]
        print(f"Bar {k:03d}: Mid=${mid:.2f} | p_dir={p_dir} | Edge={edge:.2f}b | d_macro={d_macro:+.3f} | d_meso={d_meso:+.3f} | d_tact={d_tact:+.3f}")
