import os
import glob
import json

base_dir = 'scratch'
for c in ['archive_cycle17', 'archive_cycle19', 'archive_cycle43', 'archive_cycle52', 'archive_cycle57']:
    p = os.path.join(base_dir, c)
    ledgers = glob.glob(os.path.join(p, '*ledger*.jsonl'))
    if not ledgers: continue
    with open(ledgers[0], 'r', encoding='utf-8') as f:
        bars = [json.loads(l) for l in f if l.strip().startswith('{')]
    mids = [b.get('bbo_mid', b.get('mid')) for b in bars]
    open_p = mids[0]
    close_p = mids[-1]
    hi = max(mids)
    lo = min(mids)
    ret_bps = ((close_p - open_p) / open_p) * 1e4
    rng_bps = ((hi - lo) / open_p) * 1e4
    print(f"=== {c} ===")
    print(f"Open: ${open_p:.2f} | Close: ${close_p:.2f} | Net Ret: {ret_bps:+.2f} bps | Range: {rng_bps:.2f} bps")
