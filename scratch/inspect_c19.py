import os
import glob
import json
import sys
sys.path.append('.')

arch = "D:/PrivateProjects/BTC Futures/scratch/archive_cycle19"
ledgers = glob.glob(os.path.join(arch, "*ledger*.jsonl"))
with open(ledgers[0], "r", encoding="utf-8") as f:
    bars = [json.loads(l) for l in f if l.strip().startswith("{")]

print(f"Cycle 19 has {len(bars)} bars")

from scratch.test_cycle65_with_quantile_and_wide_sl import all_closed_trades as oo_trades
c19_trades = [t for t in oo_trades if t['cycle'] == 'archive_cycle19']
print("OO trades in Cycle 19:", c19_trades)
