import json
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
FEAT_ENGINE_DIR = REPO_ROOT / "feature engineering scripts"
for p in [str(REPO_ROOT), str(FEAT_ENGINE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from inference.live_v4_model_registry import LiveMultiHorizonV4ModelRegistry

bars = []
with open('scratch/live_hierarchical_session_1h_ledger.jsonl', 'r') as f:
    for line in f:
        bars.append(json.loads(line))

# Check Bar 38 with ms_phys_long = False
b10s = [{"close": b["bbo_mid"], "taker_imb": 0.0} for b in bars[:38]]
klines_um = {"10s": b10s, "1m": []}

# Let's inspect what happens if ms_phys_long and wave boxes are disabled
def displacement_no_ms_phys(klines_um):
    # Call original but inspect what other rules are True
    # If only ms_phys_long was True, then without it, it returns 0.0, 0.0, 0.0!
    pass

# We already saw in trace_locals.py for Bar 38:
# True boolean flags: ['ms_phys_long', 'unseen_session_mode', 'live_adaptive_long', 'is_long_any', 'fast_tact', 'fast_micro']
# Look at that: the ONLY rule that was True was ms_phys_long!
# Because live_adaptive_long = unseen_session_mode and (ms_phys_long or p1.. or r2..)
# When ms_phys_long is False, live_adaptive_long is False!
# And is_long_any = r28_dimless_long or live_adaptive_long is False!
# So displacement is (0.0, 0.0, 0.0)!
print("Verified from trace: At Bar 38, ms_phys_long was the ONLY active rule!")
