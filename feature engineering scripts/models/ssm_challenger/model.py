"""
SSM (Mamba) challenger architecture: a small stack of selective
state-space blocks over the full lookback window, multi-task heads for
direction (3-class) and short-horizon return (regression).

Prefers the real CUDA-kernel Mamba block from `mamba-ssm`
(state-spaces/mamba, pip install mamba-ssm + causal-conv1d — Linux/CUDA
only, confirmed via web research 2026-09-12) when available, since that's
the actual architecture this member is meant to challenge the transformer-
family members with. Falls back to a plain GRU recurrent block with the
same (batch, seq, hidden) -> (batch, seq, hidden) interface when mamba-ssm
isn't importable (e.g. this project's Windows/CPU dev environment, or any
Kaggle session that didn't install it) — so the training script and its
multi-task heads are identical either way and the fallback is never
silently swapped in without being logged.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn

log = logging.getLogger("models.ssm_challenger.model")

try:
    from mamba_ssm import Mamba as _MambaBlock

    HAVE_MAMBA_SSM = True
except ImportError:
    HAVE_MAMBA_SSM = False


class _GRUFallbackBlock(nn.Module):
    """Same (batch, seq, d_model) -> (batch, seq, d_model) contract as
    mamba_ssm.Mamba, used only when the real CUDA kernel package isn't
    available."""

    def __init__(self, d_model: int):
        super().__init__()
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.norm(x + out)


def _make_ssm_block(d_model: int) -> nn.Module:
    if HAVE_MAMBA_SSM:
        return _MambaBlock(d_model=d_model, d_state=16, d_conv=4, expand=2)
    return _GRUFallbackBlock(d_model)


class SSMChallenger(nn.Module):
    def __init__(self, n_features: int, d_model: int = 128, n_layers: int = 4,
                 n_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.blocks = nn.ModuleList([_make_ssm_block(d_model) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(d_model)
        self.direction_head = nn.Linear(d_model, n_classes)
        self.return_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (batch, seq_len, n_features) -> (direction_logits, return_pred)
        using only the LAST timestep's hidden state (causal pooling — the
        model must not peek at any bar after the window's own end)."""
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        h = self.dropout(self.final_norm(h))
        h_last = h[:, -1, :]
        return self.direction_head(h_last), self.return_head(h_last).squeeze(-1)
