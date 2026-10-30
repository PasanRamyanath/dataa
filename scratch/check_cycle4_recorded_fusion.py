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

pfl_values = []
edge_values = []
raw_dirs = []

for b in bars:
    p_mac = np.array(b['macro_probs'], dtype=np.float64)
    p_mes = np.array(b['meso_probs'], dtype=np.float64)
    p_tac = np.array(b['tactical_probs'], dtype=np.float64)
    p_mic = np.array(b['micro_probs'], dtype=np.float64)
    
    # Calculate analytical swing as in hierarchical_fusion_v3.py
    d_macro = float(p_mac[2] - p_mac[0])
    d_meso = float(p_mes[2] - p_mes[0])
    d_tact = float(p_tac[2] - p_tac[0])
    d_micro = float(p_mic[2] - p_mic[0])
    
    macro_conflict = (d_macro * d_meso) < -0.05
    macro_gate = 0.45 if macro_conflict else 1.0
    
    raw_swing_probs = (
        0.15 * p_mac
        + 0.45 * (p_mes * macro_gate)
        + 0.30 * p_tac
        + 0.10 * p_mic
    )
    analytical_swing = raw_swing_probs / np.sum(raw_swing_probs)
    p_dn, p_fl, p_up = analytical_swing[0], analytical_swing[1], analytical_swing[2]
    edge = p_up - p_dn
    
    pfl_values.append(p_fl)
    edge_values.append(edge)

print(f"p_fl distribution: min={min(pfl_values):.4f}, mean={np.mean(pfl_values):.4f}, max={max(pfl_values):.4f}")
print(f"Bars with p_fl > 0.44: {sum(1 for x in pfl_values if x > 0.44)} / {len(bars)} ({sum(1 for x in pfl_values if x > 0.44)/len(bars)*100:.1f}%)")
print(f"Bars with abs(edge) >= 0.15: {sum(1 for x in edge_values if abs(x) >= 0.15)}")
print(f"Bars with abs(edge) >= 0.15 AND p_fl <= 0.44: {sum(1 for e, f in zip(edge_values, pfl_values) if abs(e) >= 0.15 and f <= 0.44)}")
print(f"Bars with abs(edge) >= 0.15 AND p_fl <= 0.50: {sum(1 for e, f in zip(edge_values, pfl_values) if abs(e) >= 0.15 and f <= 0.50)}")
print(f"Bars with abs(edge) >= 0.15 AND p_fl <= 0.55: {sum(1 for e, f in zip(edge_values, pfl_values) if abs(e) >= 0.15 and f <= 0.55)}")
