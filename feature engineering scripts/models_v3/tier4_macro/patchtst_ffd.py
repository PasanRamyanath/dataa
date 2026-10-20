"""
feature engineering scripts/models_v3/tier4_macro/patchtst_ffd.py
Channel-Independent & Cross-Channel PatchTST over Stationary Fractionally
Differentiated Multi-Timeframe Inputs (4h, 6h, 8h, 12h, 1d).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MacroPatchTSTModel(nn.Module):
    """
    Segments lookback length L into overlapping sub-series patches of length P
    with stride S, applies Transformer self-attention over patches per channel,
    and fuses channel representations to predict Macro Triple-Barrier outcomes.
    """

    def __init__(
        self,
        seq_len: int = 240,       # e.g. 240 bars of 4h = 40 days
        n_channels: int = 12,     # FFD close, YZ vol, OI, Basis, Funding, CVD, ETH/SOL/BNB rel strength
        patch_len: int = 12,      # 12 bars of 4h = 48 hours (2 days) per patch
        stride: int = 6,          # 6 bars of 4h = 24 hours (1 day) patch stride
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.n_channels = n_channels
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = (seq_len - patch_len) // stride + 1

        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head_norm = nn.LayerNorm(n_channels * d_model)
        self.cls_head = nn.Sequential(
            nn.Linear(n_channels * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3),
        )
        self.drift_head = nn.Linear(n_channels * d_model, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # x shape: (batch, seq_len, n_channels)
        bsz, seq_l, n_ch = x.shape
        # Per-channel standardization across lookback window
        mean = x.mean(dim=1, keepdim=True)
        std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_norm = (x - mean) / std

        # Unfold into patches: (batch, n_channels, num_patches, patch_len)
        x_ch = x_norm.transpose(1, 2)
        patches = x_ch.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # Merge (batch * n_channels) for channel-independent patch encoding
        patches_flat = patches.contiguous().view(bsz * n_ch, self.num_patches, self.patch_len)
        h = self.patch_embed(patches_flat) + self.pos_embed
        h = self.encoder(h)

        # Pool last 2 patches (most recent macro state) per channel
        h_recent = h[:, -2:, :].mean(dim=1).view(bsz, n_ch * h.shape[-1])
        h_recent = self.head_norm(h_recent)

        logits = self.cls_head(h_recent)
        probs = F.softmax(logits, dim=-1)
        macro_drift_bps = self.drift_head(h_recent).squeeze(-1)
        return {
            "logits": logits,
            "probs": probs,
            "macro_drift_bps": macro_drift_bps,
        }
