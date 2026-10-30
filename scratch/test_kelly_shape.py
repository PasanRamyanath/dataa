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

# Test with 2-element align_3_4
align_1_2 = np.array([0.5, 0.5])
# Look at line 252 of hierarchical_fusion_v3.py:
# half_kelly = 0.50 * raw_kelly * regime_vol_multiplier * max(0.25, 0.5 + 0.5 * cross_alignment)
# What is cross_alignment?
# coherence = sign_target * (0.25 * d_macro + 0.40 * d_meso + 0.25 * d_tact + 0.10 * d_micro)
# What if d_macro is a numpy array?
p_macro = np.array([0.1, 0.4, 0.5]) # shape (3,)
d_macro = float(p_macro[2] - p_macro[0]) # scalar float
print("d_macro type:", type(d_macro))

# What if raw_kelly is computed when stacker_res is present?
# Let's inspect stacker_res
if fusion.stacker is not None and getattr(fusion.stacker, "is_fitted_", False):
    res = fusion.stacker.predict_live(
        tier4_macro_probs=p_macro,
        tier3_meso_probs=p_macro,
        tier2_tactical_probs=p_macro,
        tier1_micro_probs=p_macro,
        hmm_regime_probs=np.array([0.25, 0.25, 0.25, 0.25]),
        atr_15m_bps=20.0,
        atr_1h_bps=25.0,
    )
    print("stacker_res p_long_profitable:", type(res["p_long_profitable"]), np.shape(res["p_long_profitable"]))
    print("stacker_res p_short_profitable:", type(res["p_short_profitable"]), np.shape(res["p_short_profitable"]))
    print("stacker_res calibrated_probs:", type(res["calibrated_probs"]), np.shape(res["calibrated_probs"]))
