import json
import glob

for next_arch in ['archive_cycle32', 'archive_cycle40', 'archive_cycle56']:
    f = glob.glob(f'D:/PrivateProjects/BTC Futures/scratch/{next_arch}/*ledger*.jsonl')[0]
    first_b = json.loads(open(f).readline())
    print(f"{next_arch} first bar mid: {first_b.get('bbo_mid')}")
