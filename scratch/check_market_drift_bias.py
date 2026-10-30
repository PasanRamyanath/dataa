import os
import glob
import json
import numpy as np

base_dir = 'scratch'
archives = sorted([d for d in os.listdir(base_dir) if (d.startswith('archive_cycle') or d.startswith('archive_v3')) and os.path.isdir(os.path.join(base_dir, d))])

up_sessions = []
down_sessions = []
flat_sessions = []

for arch in archives:
    ledgers = glob.glob(os.path.join(base_dir, arch, '*ledger*.jsonl'))
    if not ledgers: continue
    with open(ledgers[0], 'r', encoding='utf-8') as f:
        bars = [json.loads(l) for l in f if l.strip().startswith('{')]
    mids = [b.get('bbo_mid', b.get('mid')) for b in bars]
    ret_bps = ((mids[-1] - mids[0]) / mids[0]) * 1e4
    if ret_bps > 10.0:
        up_sessions.append((arch, ret_bps))
    elif ret_bps < -10.0:
        down_sessions.append((arch, ret_bps))
    else:
        flat_sessions.append((arch, ret_bps))

print(f"Total Sessions: {len(archives)}")
print(f"UP Sessions (> +10 bps):   {len(up_sessions)} ({len(up_sessions)/len(archives)*100:.1f}%) | Avg return: {np.mean([r for _, r in up_sessions]):+.1f} bps")
print(f"DOWN Sessions (< -10 bps): {len(down_sessions)} ({len(down_sessions)/len(archives)*100:.1f}%) | Avg return: {np.mean([r for _, r in down_sessions]):+.1f} bps")
print(f"FLAT Sessions (-10 to +10):{len(flat_sessions)} ({len(flat_sessions)/len(archives)*100:.1f}%) | Avg return: {np.mean([r for _, r in flat_sessions]):+.1f} bps")
