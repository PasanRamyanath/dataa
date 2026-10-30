import os
import glob
import json
import collections

base_dir = "D:/PrivateProjects/BTC Futures/scratch"
archives = sorted([d for d in os.listdir(base_dir) if (d.startswith("archive_cycle") or d.startswith("archive_v3")) and os.path.isdir(os.path.join(base_dir, d))])

from scratch.test_cycle65_with_quantile_and_wide_sl import all_closed_trades as oo_trades

# Find the 7 losses in test_continuous_trade_completion
from scratch.test_continuous_trade_completion import archives, base_dir
# Let's inspect which 2 cycles had different losses
oo_losses = [t['cycle'] for t in oo_trades if t['net'] <= 0]
print("OO losses:", sorted(oo_losses))
