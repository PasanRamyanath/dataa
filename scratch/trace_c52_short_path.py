import os
import glob
import json

base_dir = 'scratch'
path = os.path.join(base_dir, 'archive_cycle52')
ledgers = glob.glob(os.path.join(path, '*ledger*.jsonl'))
with open(ledgers[0], 'r', encoding='utf-8') as f:
    bars = [json.loads(l) for l in f if l.strip().startswith('{')]

mids = [b.get('bbo_mid', b.get('mid')) for b in bars]
entry_p = 79871.90
open_p = mids[0]

print(f"Cycle 52 Open: ${open_p:.2f} | Entry at Bar 63: ${entry_p:.2f}")
for idx in range(63, min(len(bars), 215)):
    mid = mids[idx]
    b = bars[idx]
    elapsed = (idx - 63) * 10
    unreal = -((mid - entry_p) / entry_p) * 1e4
    drift_open = ((mid - open_p) / open_p) * 1e4
    p_dir = b.get('primary_direction', 1)
    tact = b.get('tactical_probs', [0.33, 0.33, 0.33])
    meso = b.get('meso_probs', [0.33, 0.33, 0.33])
    macro = b.get('macro_probs', [0.33, 0.33, 0.33])
    d_tact = tact[2] - tact[0]
    d_meso = meso[2] - meso[0]
    d_macro = macro[2] - macro[0]
    if idx in (63, 70, 80, 90, 100, 120, 140, 153, 170, 190, 208):
        print(f"Bar {idx:03d} (t+{elapsed}s): Mid=${mid:.2f} | Unreal={unreal:+.2f}b | DriftOpen={drift_open:+.2f}b | p_dir={p_dir} | d_tact={d_tact:+.2f} | d_meso={d_meso:+.2f} | d_macro={d_macro:+.2f}")
