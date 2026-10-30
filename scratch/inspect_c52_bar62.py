import sys, os, glob, json
sys.stdout.reconfigure(encoding='utf-8')
arch = 'archive_cycle52'
ledgers = glob.glob(os.path.join('scratch', arch, '*ledger*.jsonl'))
with open(ledgers[0], 'r', encoding='utf-8') as f:
    bars = [json.loads(l) for l in f if l.strip().startswith('{')]

mids = [b.get('bbo_mid', b.get('mid', 0.0)) for b in bars]
for b in range(55, 66):
    bar = bars[b]
    mid = mids[b]
    sub = mids[:b+1]
    rng = max(sub) - min(sub)
    pct = (mid - min(sub)) / rng if rng > 0 else 0.5
    print(f"Bar {b}: mid={mid:.2f}, p_dir={bar.get('primary_direction')}, edge={bar.get('expected_net_edge_bps'):.1f}, pct_of_range={pct:.2f}, tact={bar.get('tactical_probs')}, meso={bar.get('meso_probs')}, macro={bar.get('macro_probs')}")
