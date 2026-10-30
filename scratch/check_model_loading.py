import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FEAT_ENGINE_DIR = REPO_ROOT / "feature engineering scripts"
for p in [str(REPO_ROOT), str(FEAT_ENGINE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from inference.live_v4_model_registry import LiveMultiHorizonV4ModelRegistry

reg = LiveMultiHorizonV4ModelRegistry()
print("=== MODEL CHECKPOINT LOADING STATUS ===")
print("tier4_hmm:", reg.tier4_hmm is not None)
print("tier4_patchtst:", reg.tier4_patchtst is not None)
print("tier4_tft:", reg.tier4_tft is not None)
print("tier3_itransformer:", reg.tier3_itransformer is not None)
print("tier3_nhits:", reg.tier3_nhits is not None)
print("tier3_lgbm:", reg.tier3_lgbm is not None)
print("tier2_mamba_model:", reg.tier2_mamba_model is not None)
print("tier2_lgbm_bookdepth:", reg.tier2_lgbm_bookdepth is not None)
print("tier2_crossasset_gat:", reg.tier2_crossasset_gat is not None)
print("tier2_cryptobert_news:", reg.tier2_cryptobert_news is not None)
print("tier1_lgbm_micro:", reg.tier1_lgbm_micro is not None)
print("tier1_spot1s_leadlag:", reg.tier1_spot1s_leadlag is not None)
print("tier1_queue_hazard:", reg.tier1_queue_hazard is not None)
print("survival_hazard_model:", reg.survival_hazard_model is not None)
