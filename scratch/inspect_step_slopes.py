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

prices = np.array([b["bbo_mid"] for b in bars], dtype=np.float64)
reg = LiveMultiHorizonV4ModelRegistry()

# Test iTransformer on closes
if reg.tier3_itransformer:
    for window in [15, 30, 60, 120]:
        c_sub = prices[-window:]
        p_raw = reg.tier3_itransformer.predict_from_series(c_sub, aux_bias=0.0)
        print(f"iTransformer (window={window:3d}): {p_raw}")

if reg.tier4_patchtst:
    for window in [15, 30, 60, 120]:
        c_sub = prices[-window:]
        p_raw = reg.tier4_patchtst.predict_from_series(c_sub, aux_bias=0.0)
        print(f"PatchTST     (window={window:3d}): {p_raw}")
