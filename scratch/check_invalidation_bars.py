import json
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
FEAT_ENGINE_DIR = REPO_ROOT / "feature engineering scripts"
for p in [str(REPO_ROOT), str(FEAT_ENGINE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from fusion.hierarchical_fusion_v3 import HierarchicalMultiHorizonFusionEngine

bars = [json.loads(l) for l in open('scratch/live_hierarchical_session_1h_ledger.jsonl')]
fusion = HierarchicalMultiHorizonFusionEngine()

for i in range(255, 275):
    b = bars[i]
    p_mac = np.array(b['macro_probs'], dtype=np.float64)
    p_mes = np.array(b['meso_probs'], dtype=np.float64)
    p_tac = np.array(b['tactical_probs'], dtype=np.float64)
    p_mic = np.array(b['micro_probs'], dtype=np.float64)
    
    dec = fusion.evaluate_multi_horizon_tick(
        ts_ms=b['ts_ms'],
        p_macro=p_mac,
        p_meso=p_mes,
        p_tactical=p_tac,
        p_micro=p_mic,
        atr_15m_bps=20.0,
        atr_1h_bps=25.0,
        atr_10s_bps=3.0,
    )
    print(f"Bar {i}: mid={b['bbo_mid']:.2f} | dir={dec.primary_direction} (0=S, 1=F, 2=L) | edge_bps={dec.expected_net_edge_bps} | meta={dec.meta_act_probability} | cross={dec.cross_horizon_alignment}")
