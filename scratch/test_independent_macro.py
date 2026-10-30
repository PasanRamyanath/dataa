import json
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
FEAT_ENGINE_DIR = REPO_ROOT / "feature engineering scripts"
for p in [str(REPO_ROOT), str(FEAT_ENGINE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from inference.live_v4_model_registry import LiveMultiHorizonV4ModelRegistry, _score_to_simplex
from streaming.multi_horizon_ring_buffer import MultiHorizonLiveRingBuffer

bars = []
with open('scratch/live_hierarchical_session_1h_ledger.jsonl', 'r') as f:
    for line in f:
        bars.append(json.loads(line))

rb = MultiHorizonLiveRingBuffer()
# Bootstrap real 1h/4h klines from Binance
counts = rb.bootstrap_from_binance_rest(limit=100)
print(f"Bootstrapped klines: {counts}")

reg = rb.model_registry

# Test Tier 4 Macro without any meso_anchor contamination
print("\n--- Testing Independent Tier 4 Macro ---")
closes_1h = np.array([b["close"] for b in rb.klines_um.get("1h", [])], dtype=np.float64)
closes_4h = np.array([b["close"] for b in rb.klines_um.get("4h", [])], dtype=np.float64)
print(f"Closes 1h count: {len(closes_1h)}, last 5: {closes_1h[-5:]}")
print(f"Closes 4h count: {len(closes_4h)}, last 5: {closes_4h[-5:]}")

# HMM regime
hmm_post, p_hmm = reg.predict_hmm_regime(closes_1h)
print(f"HMM Regime probs: {hmm_post}")
print(f"HMM Directional:  {p_hmm}")

# PatchTST without fake bias
p_ptst = reg.tier4_patchtst.predict_from_series(closes_4h, aux_bias=0.0)
print(f"PatchTST (bias=0.0): {p_ptst}")

# TFT without fake bias
p_tft = reg.tier4_tft.predict_from_series(closes_1h, aux_bias=0.0)
print(f"TFT (bias=0.0):      {p_tft}")
