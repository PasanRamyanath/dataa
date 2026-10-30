import os
import glob
import json

base_dir = 'scratch'
cycles = ['archive_cycle17', 'archive_cycle19', 'archive_cycle43', 'archive_cycle52', 'archive_cycle57']

for c in cycles:
    p = os.path.join(base_dir, c)
    ledgers = glob.glob(os.path.join(p, '*ledger*.jsonl'))
    if not ledgers: continue
    with open(ledgers[0], 'r', encoding='utf-8') as f:
        bars = [json.loads(l) for l in f if l.strip().startswith('{')]
    mids = [b.get('bbo_mid', b.get('mid')) for b in bars]
    
    # check first 50 bars
    print(f"\n==================== {c} First 30 Bars Analysis ====================")
    for i in range(min(30, len(bars))):
        b = bars[i]
        mid = mids[i]
        tact = b.get('tactical_probs', [0.33, 0.33, 0.33])
        meso = b.get('meso_probs', [0.33, 0.33, 0.33])
        macro = b.get('macro_probs', [0.33, 0.33, 0.33])
        p_dir = b.get('primary_direction', b.get('direction', 1))
        edge = b.get('expected_net_edge_bps', 0.0)
        d_tact = tact[2] - tact[0]
        d_meso = meso[2] - meso[0]
        d_macro = macro[2] - macro[0]
        ret_open = ((mid - mids[0]) / mids[0]) * 1e4
        if i in (0, 5, 10, 14, 15, 18, 20, 25):
            print(f"Bar {i:02d}: Mid=${mid:.2f} | RetOpen={ret_open:+.2f}b | p_dir={p_dir} | Edge={edge:.2f}b | d_tact={d_tact:+.2f} | d_meso={d_meso:+.2f} | d_macro={d_macro:+.2f}")
