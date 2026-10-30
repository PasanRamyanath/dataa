import sys, os, glob, json
sys.stdout.reconfigure(encoding='utf-8')
arch = 'archive_cycle52'
ledgers = glob.glob(os.path.join('scratch', arch, '*ledger*.jsonl'))
with open(ledgers[0], 'r', encoding='utf-8') as f:
    bars = [json.loads(l) for l in f if l.strip().startswith('{')]

for b in range(60, 210, 10):
    mid = bars[b].get('bbo_mid', bars[b].get('mid', 0.0))
    drift = ((mid - bars[0].get('bbo_mid', bars[0].get('mid', 0.0))) / bars[0].get('bbo_mid', bars[0].get('mid', 0.0))) * 1e4
    print(f"Bar {b:3d}: mid={mid:.2f}, drift={drift:+.2f}b, p_dir={bars[b].get('primary_direction')}")
