import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
FEAT_ENGINE_DIR = REPO_ROOT / "feature engineering scripts"
for p in [str(REPO_ROOT), str(FEAT_ENGINE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from fusion.hierarchical_fusion_v3 import HierarchicalMultiHorizonFusionEngine

fusion = HierarchicalMultiHorizonFusionEngine()

# Test evaluate_multi_horizon_tick with primary_dir = 2
p_macro = np.array([0.10, 0.30, 0.60])
p_meso = np.array([0.10, 0.30, 0.60])
p_tactical = np.array([0.10, 0.30, 0.60])
p_micro = np.array([0.10, 0.30, 0.60])

try:
    dec = fusion.evaluate_multi_horizon_tick(
        ts_ms=1000000,
        p_macro=p_macro,
        p_meso=p_meso,
        p_tactical=p_tactical,
        p_micro=p_micro,
        atr_15m_bps=20.0,
        atr_1h_bps=25.0,
        atr_10s_bps=3.0,
        micro_safe_fill_prob=0.85,
        hmm_regime_probs=np.array([0.25, 0.25, 0.25, 0.25]),
    )
    print("Success! Mode:", dec.execution_mode, "Kelly:", dec.kelly_position_fraction)
except Exception as e:
    import traceback
    traceback.print_exc()
