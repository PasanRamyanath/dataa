"""
SSM (Mamba) challenger architecture: a stack of selective state-space blocks
over the full lookback window, multi-task heads for direction (3-class) and
short-horizon return (regression).

In v4.0:
  - Uses `mamba_ssm.Mamba` CUDA kernel when installed on Linux/CUDA.
  - Otherwise uses `FastParallelSelectiveSSMCore` (O(log2 L) parallel associative
    prefix scan S6) for new models, while retaining `_GRUFallbackBlock` compatibility
    when loading legacy checkpoints trained with GRU keys.
"""
from __future__ import annotations

import logging
import torch
import torch.nn as nn

from models_v3.tier2_tactical.parallel_scan_s6 import FastParallelSelectiveSSMCore

log = logging.getLogger("models.ssm_challenger.model")

try:
    from mamba_ssm import Mamba as _MambaBlock
    HAVE_MAMBA_SSM = True
except ImportError:
    HAVE_MAMBA_SSM = False


class _GRUFallbackBlock(nn.Module):
    """Legacy GRU fallback block preserved for backward-compatible state_dict loading."""

    def __init__(self, d_model: int):
        super().__init__()
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.norm(x + out)


class _ParallelS6Block(nn.Module):
    """Pure-PyTorch O(log2 L) Parallel Associative Scan S6 Mamba Block (zero nvcc build needed)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.s6 = FastParallelSelectiveSSMCore(d_inner=d_model, d_state=16, dt_rank=16)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.s6(x))


def _make_ssm_block(d_model: int, legacy_gru: bool = True) -> nn.Module:
    if HAVE_MAMBA_SSM and not legacy_gru:
        return _MambaBlock(d_model=d_model, d_state=16, d_conv=4, expand=2)
    if legacy_gru or not HAVE_MAMBA_SSM:
        return _GRUFallbackBlock(d_model)
    return _ParallelS6Block(d_model)


class SSMChallenger(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        n_layers: int = 4,
        n_classes: int = 3,
        dropout: float = 0.1,
        legacy_gru: bool = True,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.input_proj = nn.Linear(n_features, d_model)
        self.blocks = nn.ModuleList([_make_ssm_block(d_model, legacy_gru=legacy_gru) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(d_model)
        self.direction_head = nn.Linear(d_model, n_classes)
        self.return_head = nn.Linear(d_model, 1)

    def load_state_dict(self, state_dict, strict: bool = True):
        # Auto-detect if checkpoint was saved with legacy _GRUFallbackBlock keys
        has_gru_keys = any(".gru." in k for k in state_dict.keys())
        has_s6_block = any(isinstance(b, _ParallelS6Block) for b in self.blocks)
        if has_gru_keys and has_s6_block:
            self.blocks = nn.ModuleList([_GRUFallbackBlock(self.d_model) for _ in range(self.n_layers)])
            self.blocks.to(self.input_proj.weight.device)
        return super().load_state_dict(state_dict, strict=strict)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        h = self.dropout(self.final_norm(h))
        h_last = h[:, -1, :]
        return self.direction_head(h_last), self.return_head(h_last).squeeze(-1)
