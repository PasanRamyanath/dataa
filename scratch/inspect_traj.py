import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

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

# Inspect iTransformer tensors and traj
runner = reg.tier3_itransformer
c = np.asarray(prices[-60:], dtype=np.float32)
c_mean = float(np.mean(c))
c_std = max(float(np.std(c)), c_mean * 0.0005, 1e-4)
c_norm = (c - c_mean) / c_std

with torch.inference_mode():
    w_emb = runner.tensors["enc_embedding.value_embedding.weight"]
    b_emb = runner.tensors["enc_embedding.value_embedding.bias"]
    w_proj = runner.tensors["projector.weight"]
    b_proj = runner.tensors["projector.bias"]
    seq_len = w_emb.shape[1]
    x_pad = np.pad(c_norm[-seq_len:], (max(0, seq_len - len(c_norm)), 0), mode="edge")
    x_t = torch.from_numpy(x_pad).view(1, seq_len)
    h = F.gelu(F.linear(x_t, w_emb, b_emb))
    if "encoder.norm.weight" in runner.tensors:
        h = F.layer_norm(h, (h.shape[-1],), runner.tensors["encoder.norm.weight"], runner.tensors["encoder.norm.bias"])
    traj = F.linear(h, w_proj, b_proj).view(-1).numpy()

print(f"iTransformer traj shape: {traj.shape}")
print(f"iTransformer traj[:10]: {traj[:10]}")
step_slope = float(0.50 * (traj[1] - traj[0]) + 0.30 * (traj[min(3, len(traj)-1)] - traj[0]) + 0.20 * (traj[-1] - traj[0]))
print(f"step_slope: {step_slope}")
