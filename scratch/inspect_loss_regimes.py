import os
import glob
import json
import sys

base_dir = 'scratch'
cycles = ['archive_cycle17', 'archive_cycle40', 'archive_cycle43', 'archive_cycle52', 'archive_cycle57', 'archive_cycle58']

for c in cycles:
    p = os.path.join(base_dir, c)
    ledgers = glob.glob(os.path.join(p, '*ledger*.jsonl'))
    if not ledgers: continue
    with open(ledgers[0], 'r', encoding='utf-8') as f:
        bars = [json.loads(l) for l in f if l.strip().startswith('{')]
    
    mids = [b.get('bbo_mid', b.get('mid')) for b in bars]
    high = max(mids)
    low = min(mids)
    open_p = mids[0]
    close_p = mids[-1]
    sess_ret_bps = ((close_p - open_p) / open_p) * 1e4
    sess_rng_bps = ((high - low) / open_p) * 1e4
    
    print(f"=== {c} ===")
    print(f"Session Open: ${open_p:.2f} | Close: ${close_p:.2f} | High: ${high:.2f} | Low: ${low:.2f}")
    print(f"Session Return: {sess_ret_bps:+.2f} bps | Session Range: {sess_rng_bps:.2f} bps")
