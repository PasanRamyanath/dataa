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

from inference.live_v4_model_registry import LiveMultiHorizonV4ModelRegistry, _score_to_simplex

bars = []
with open('scratch/live_hierarchical_session_1h_ledger.jsonl', 'r') as f:
    for line in f:
        bars.append(json.loads(line))

prices = np.array([b["bbo_mid"] for b in bars], dtype=np.float64)
reg = LiveMultiHorizonV4ModelRegistry()

for name, runner in [
    ("iTransformer", reg.tier3_itransformer),
    ("PatchTST", reg.tier4_patchtst),
    ("NHITS", reg.tier3_nhits),
    ("TFT", reg.tier4_tft),
]:
    if runner is None:
        continue
    c = np.asarray(prices[-60:], dtype=np.float32)
    c_mean = float(np.mean(c))
    c_std = max(float(np.std(c)), c_mean * 0.0005, 1e-4)
    c_norm = (c - c_mean) / c_std

    with torch.inference_mode():
        if runner.arch == "itransformer":
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
        elif runner.arch == "patchtst":
            w_in = runner.tensors["model.backbone.input_layer.weight"]
            b_in = runner.tensors["model.backbone.input_layer.bias"]
            w_head = runner.tensors["model.head.linear.weight"]
            b_head = runner.tensors["model.head.linear.bias"]
            patch_len = w_in.shape[1]
            d_model = w_in.shape[0]
            n_patches = w_head.shape[1] // d_model
            stride = 8
            req_len = (n_patches - 1) * stride + patch_len
            x_pad = np.pad(c_norm[-req_len:], (max(0, req_len - len(c_norm)), 0), mode="edge")
            x_t = torch.from_numpy(x_pad)
            patches = x_t.unfold(0, patch_len, stride)[:n_patches]
            h = F.gelu(F.linear(patches, w_in, b_in)).reshape(1, -1)
            traj = F.linear(h, w_head, b_head).view(-1).numpy()
        elif runner.arch == "nhits":
            w0 = runner.tensors["blocks.0.layers.0.weight"]
            b0 = runner.tensors["blocks.0.layers.0.bias"]
            in_dim = w0.shape[1]
            x_pad = np.pad(c_norm[-in_dim:], (max(0, in_dim - len(c_norm)), 0), mode="edge")
            h = torch.from_numpy(x_pad).view(1, in_dim)
            for l_idx in range(8):
                wk = f"blocks.0.layers.{l_idx}.weight"
                bk = f"blocks.0.layers.{l_idx}.bias"
                if wk in runner.tensors:
                    w_l, b_l = runner.tensors[wk], runner.tensors[bk]
                    if h.shape[-1] == w_l.shape[1]:
                        h = F.relu(F.linear(h, w_l, b_l)) if l_idx < 7 else F.linear(h, w_l, b_l)
            traj = h.view(-1).numpy()
        else:
            w_grn = runner.tensors.get("temporal_fusion_decoder.positionwise_grn.lin_i.weight")
            b_grn = runner.tensors.get("temporal_fusion_decoder.positionwise_grn.lin_i.bias")
            if w_grn is not None and b_grn is not None:
                in_dim = w_grn.shape[1]
                x_pad = np.pad(c_norm[-in_dim:], (max(0, in_dim - len(c_norm)), 0), mode="edge")
                h = F.elu(F.linear(torch.from_numpy(x_pad).view(1, in_dim), w_grn, b_grn)).view(-1).numpy()
                traj = h
            else:
                traj = c_norm[-8:]

    step_slope = float(0.50 * (traj[1] - traj[0]) + 0.30 * (traj[min(3, len(traj)-1)] - traj[0]) + 0.20 * (traj[-1] - traj[0])) if len(traj) >= 4 else 0.0
    print(f"Model: {name:15s} | raw step_slope: {step_slope:+.4f}")
