import os
import glob
import json
import sys
import numpy as np

base_dir = 'scratch'
cycles = ['archive_cycle17', 'archive_cycle40', 'archive_cycle43', 'archive_cycle52', 'archive_cycle57', 'archive_cycle58']

for c in cycles:
    p = os.path.join(base_dir, c)
    ledgers = glob.glob(os.path.join(p, '*ledger*.jsonl'))
    if not ledgers: continue
    with open(ledgers[0], 'r', encoding='utf-8') as f:
        bars = [json.loads(l) for l in f if l.strip().startswith('{')]
        
    mids = [b.get('bbo_mid', b.get('mid')) for b in bars]
    
    # We know the entry bar indices from earlier:
    # cycle17: bar 14 (SELL)
    # cycle40: bar 18 (BUY)
    # cycle43: bar 166 (SELL)
    # cycle52: bar 48 (SELL)
    # cycle57: bar 134 (SELL)
    # cycle58: bar 58 (BUY)
    
    entry_indices = {
        'archive_cycle17': (14, 'SELL'),
        'archive_cycle40': (18, 'BUY'),
        'archive_cycle43': (166, 'SELL'),
        'archive_cycle52': (48, 'SELL'),
        'archive_cycle57': (134, 'SELL'),
        'archive_cycle58': (58, 'BUY'),
    }
    
    idx, side = entry_indices[c]
    bar = bars[idx]
    mid = mids[idx]
    
    # Range up to entry
    seen_mids = mids[:idx+1]
    hi = max(seen_mids)
    lo = min(seen_mids)
    rng = hi - lo
    pct_rng = (mid - lo) / rng if rng > 0 else 0.5
    
    # Momentum 5 bars, 10 bars, 20 bars
    ret_5 = ((mid - mids[max(0, idx-5)]) / mids[max(0, idx-5)]) * 1e4
    ret_15 = ((mid - mids[max(0, idx-15)]) / mids[max(0, idx-15)]) * 1e4
    
    # Session open return
    ret_open = ((mid - mids[0]) / mids[0]) * 1e4
    
    # ATR 14
    atrs = [abs(mids[k] - mids[k-1]) for k in range(max(1, idx-14), idx+1)]
    atr = np.mean(atrs) if atrs else 0.0
    atr_bps = (atr / mid) * 1e4
    
    tact = bar.get('tactical_probs', [0.33, 0.33, 0.33])
    meso = bar.get('meso_probs', [0.33, 0.33, 0.33])
    macro = bar.get('macro_probs', [0.33, 0.33, 0.33])
    micro = bar.get('micro_probs', [0.33, 0.33, 0.33])
    p_dir = bar.get('primary_direction', bar.get('direction', 1))
    edge = bar.get('expected_net_edge_bps', bar.get('edge', 0.0))
    
    print(f"=== {c} at Bar {idx} ({side}) ===")
    print(f"Mid: ${mid:.2f} | Open: ${mids[0]:.2f} | Ret from Open: {ret_open:+.2f} bps")
    print(f"Range: Lo=${lo:.2f}, Hi=${hi:.2f}, Rng=${rng:.2f} ({rng/mid*1e4:.1f} bps) | Pct of Range: {pct_rng*100:.1f}%")
    print(f"Momentum: 5-bar Ret={ret_5:+.2f} bps, 15-bar Ret={ret_15:+.2f} bps | ATR(14)={atr:.2f} ({atr_bps:.2f} bps)")
    print(f"Signals: p_dir={p_dir}, Edge={edge:.2f}b")
    print(f"  Tact: d={tact[2]-tact[0]:+.3f} | Meso: d={meso[2]-meso[0]:+.3f} | Macro: d={macro[2]-macro[0]:+.3f} | Micro: d={micro[2]-micro[0]:+.3f}")
    print()
