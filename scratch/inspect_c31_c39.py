import os
import glob
import json
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

for arch in ['archive_cycle31', 'archive_cycle39']:
    base_dir = f"D:/PrivateProjects/BTC Futures/scratch/{arch}"
    ledgers = glob.glob(os.path.join(base_dir, "*ledger*.jsonl"))
    with open(ledgers[0], "r", encoding="utf-8") as f:
        bars = [json.loads(line) for line in f if line.strip().startswith("{")]

    print(f"\n{'='*60}\n{arch} (Total Bars: {len(bars)})\n{'='*60}")
    open_mid = bars[0].get('bbo_mid')
    for i in range(120, len(bars), 20):
        b = bars[i]
        mid = b.get('bbo_mid')
        drift = (mid - open_mid) / open_mid * 1e4
        tact = b.get('tactical_probs')
        meso = b.get('meso_probs')
        macro = b.get('macro_probs')
        p_dir = b.get('primary_direction')
        print(f"Bar {i:3d}: mid={mid:.2f} drift={drift:+.2f}b dir={p_dir} tact={tact} meso={meso} macro={macro}")
