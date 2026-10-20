"""
feature engineering scripts/models_v3/tier2_tactical/mamba_selective_ssm.py
Selective State-Space Model (S6 / Mamba, Gu & Dao 2023) for Tier 2 Tactical
(1m, 3m, 5m) Multi-Resolution Sequences (L = 720 to 1,440 bars).
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSMCore(nn.Module):
    """
    Exact Selective State-Space (S6) layer with HiPPO-LegS diagonal initialization
    and input-dependent step size Delta_t, B_t, and C_t projections.
    """

    def __init__(self, d_inner: int, d_state: int = 16, dt_rank: int = 16) -> None:
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank

        # HiPPO-inspired diagonal state matrix A < 0 parameterized in log-space
        a_init = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(a_init))
        self.D = nn.Parameter(torch.ones(d_inner))

        # Input-dependent projections for Delta, B, C
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # Initialize dt_proj bias so softplus(dt_bias) lies in [0.001, 0.1]
        dt = torch.exp(torch.rand(d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: Tensor of shape (batch, seq_len, d_inner)
        Returns:
            y: Tensor of shape (batch, seq_len, d_inner)
        """
        bsz, seq_len, d_in = u.shape
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state), strictly negative

        proj = self.x_proj(u)  # (B, L, dt_rank + 2*d_state)
        dt_raw, B_t, C_t = torch.split(proj, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt_raw))  # (B, L, d_inner) > 0

        # Discretize via Zero-Order Hold (ZOH): dA = exp(delta * A), dB = delta * B_t * u
        # Chunked recurrence over sequence for O(L) memory efficiency
        h = torch.zeros(bsz, d_in, self.d_state, device=u.device, dtype=u.dtype)
        ys = []
        for t in range(seq_len):
            dt_t = delta[:, t, :].unsqueeze(-1)            # (B, d_inner, 1)
            b_t = B_t[:, t, :].unsqueeze(1)                # (B, 1, d_state)
            c_t = C_t[:, t, :].unsqueeze(1)                # (B, 1, d_state)
            u_t = u[:, t, :].unsqueeze(-1)                 # (B, d_inner, 1)

            dA = torch.exp(dt_t * A.unsqueeze(0))          # (B, d_inner, d_state)
            dB_u = dt_t * b_t * u_t                        # (B, d_inner, d_state)
            h = dA * h + dB_u
            y_t = torch.sum(h * c_t, dim=-1) + self.D * u[:, t, :]
            ys.append(y_t)

        return torch.stack(ys, dim=1)


class TacticalMambaBlock(nn.Module):
    """Full Mamba block with 1D Depthwise Causal Convolution + SiLU Gating + Selective S6."""

    def __init__(self, d_model: int = 128, d_state: int = 16, expand: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        d_inner = d_model * expand
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=4,
            padding=3,
            groups=d_inner,
        )
        self.ssm = SelectiveSSMCore(d_inner=d_inner, d_state=d_state)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        residual = x
        x_norm = self.norm(x)
        xz = self.in_proj(x_norm)
        x_branch, z_gate = xz.chunk(2, dim=-1)

        # Causal 1D convolution (truncate trailing padding so zero future leakage!)
        seq_len = x_branch.shape[1]
        x_conv = self.conv1d(x_branch.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        x_act = F.silu(x_conv)

        y_ssm = self.ssm(x_act)
        y_gated = y_ssm * F.silu(z_gate)
        return residual + self.drop(self.out_proj(y_gated))


class MultiScaleTacticalMambaClassifier(nn.Module):
    """
    Tier 2 Tactical (1m, 3m, 5m) Mamba Classifier & Pullback Inflection Detector.
    """

    def __init__(
        self,
        n_features: int = 48,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 16,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
        )
        self.blocks = nn.ModuleList([
            TacticalMambaBlock(d_model=d_model, d_state=d_state, expand=2, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.direction_head = nn.Linear(d_model, 3)
        self.pullback_exhaustion_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.input_proj(x)
        for blk in self.blocks:
            h = blk(h)
        h_last = self.final_norm(h[:, -1, :])
        logits = self.direction_head(h_last)
        probs = F.softmax(logits, dim=-1)
        pullback_score = torch.sigmoid(self.pullback_exhaustion_head(h_last)).squeeze(-1)
        return {
            "logits": logits,
            "probs": probs,
            "pullback_exhaustion_score": pullback_score,
            "embedding": h_last,
        }
