import json
import sys, os
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath('feature engineering scripts'))

from execution.champion_challenger_framework import ChampionChallengerTournamentManager
from fusion.hierarchical_fusion_v3 import HierarchicalDecisionOutput

# Load Cycle 59 bars
records = []
with open('scratch/archive_cycle59/live_hierarchical_session_1h_ledger.jsonl') as f:
    for line in f:
        records.append(json.loads(line))

print(f"Loaded {len(records)} bars from Cycle 59.")

# Let's inspect each bar's range percentile
rolling_window = []
for r in records:
    mid = r["bbo_mid"]
    rolling_window.append(mid)
    if len(rolling_window) > 90:
        rolling_window.pop(0)
    
    if r["bar_idx"] in (56, 70, 129, 280):
        low = min(rolling_window)
        high = max(rolling_window)
        pct = (mid - low) / max(high - low, 1e-4) * 100.0
        print(f"Bar #{r['bar_idx']}: Mid={mid:.2f}, Low={low:.2f}, High={high:.2f}, Range={high-low:.2f}, Pct={pct:.1f}%")
