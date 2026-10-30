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
        
    macro_diffs = []
    meso_diffs = []
    tact_diffs = []
    for b in bars:
        macro = b.get('macro_probs', [0.33, 0.33, 0.33])
        meso = b.get('meso_probs', [0.33, 0.33, 0.33])
        tact = b.get('tactical_probs', [0.33, 0.33, 0.33])
        macro_diffs.append(macro[2] - macro[0])
        meso_diffs.append(meso[2] - meso[0])
        tact_diffs.append(tact[2] - tact[0])
        
    print(f"=== {c} ===")
    print(f"Mean Macro Diff: {np.mean(macro_diffs):+.4f} (min={np.min(macro_diffs):+.4f}, max={np.max(macro_diffs):+.4f})")
    print(f"Mean Meso Diff:  {np.mean(meso_diffs):+.4f} (min={np.min(meso_diffs):+.4f}, max={np.max(meso_diffs):+.4f})")
    print(f"Mean Tact Diff:  {np.mean(tact_diffs):+.4f} (min={np.min(tact_diffs):+.4f}, max={np.max(tact_diffs):+.4f})")
