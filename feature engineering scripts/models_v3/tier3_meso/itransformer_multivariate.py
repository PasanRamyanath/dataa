"""
feature engineering scripts/models_v3/tier3_meso/itransformer_multivariate.py
True Multivariate Inverted Transformer (Liu et al., ICLR 2024, arXiv:2310.06625)
operating across N=32 synchronized multi-timeframe (15m, 30m, 1h, 2h, 4h) and
cross-asset (BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, Spot, Perp, OI, BVOL, Funding) variates.
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MESO_32_VARIATE_MANIFEST: list[str] = [
    # 1-4: USD(S)-M 15m FFD (d=0.35) Log-Prices
    "btcusdt_um_15m_ffd035",
    "ethusdt_um_15m_ffd035",
    "solusdt_um_15m_ffd035",
    "bnbusdt_um_15m_ffd035",
    # 5-8: Spot 15m FFD (d=0.35) Log-Prices
    "btcusdt_spot_15m_ffd035",
    "ethusdt_spot_15m_ffd035",
    "solusdt_spot_15m_ffd035",
    "bnbusdt_spot_15m_ffd035",
    # 9-12: Multi-Timeframe Spot-Perp Basis (bps)
    "btcusdt_basis_15m_bps",
    "btcusdt_basis_30m_bps",
    "btcusdt_basis_1h_bps",
    "btcusdt_basis_4h_bps",
    # 13-16: Cross-Asset 15m Taker Buy/Sell Volume Imbalance
    "btcusdt_um_15m_taker_imb",
    "ethusdt_um_15m_taker_imb",
    "solusdt_um_15m_taker_imb",
    "bnbusdt_um_15m_taker_imb",
    # 17-20: Binance Futures 5m Metrics (Resampled to 15m Close)
    "btcusdt_oi_delta_15m_pct",
    "btcusdt_toptrader_pos_ls_ratio",
    "btcusdt_toptrader_acc_ls_ratio",
    "btcusdt_retail_ls_ratio",
    # 21-24: Premium Index & 8h Funding Pressure
    "btcusdt_premium_15m_bps",
    "btcusdt_premium_1h_bps",
    "btcusdt_premium_4h_bps",
    "btcusdt_funding_8h_zscore",
    # 25-28: Volatility Term Structure & Options BVOL VRP
    "btcbvol_implied_vol_15m",
    "btcusdt_yz_rvol_15m",
    "btcusdt_yz_rvol_1h",
    "btcusdt_vrp_15m_spread",
    # 29-32: COIN-M Basis, Quarterly Roll Yield, BookDepth & News Sentiment
    "btcusd_cm_basis_15m_bps",
    "btcusdt_quarterly_roll_yield",
    "btcusdt_bookdepth_2pct_imb",
    "crypto_news_sentiment_15m",
]


class ReversibleInstanceNorm1d(nn.Module):
    """RevIN (Kim et al., ICLR 2022) per-variate temporal standardization."""

    def __init__(self, num_variates: int, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        self.num_variates = num_variates
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, num_variates))
            self.beta = nn.Parameter(torch.zeros(1, 1, num_variates))

    def forward(self, x: torch.Tensor, mode: str = "norm") -> torch.Tensor:
        # x shape: (batch, seq_len, num_variates)
        if mode == "norm":
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            x_norm = (x - self.mean) / self.stdev
            if self.affine:
                x_norm = x_norm * self.gamma + self.beta
            return x_norm
        elif mode == "denorm":
            if self.affine:
                x = (x - self.beta) / (self.gamma + self.eps)
            return x * self.stdev + self.mean
        raise ValueError(f"Unsupported RevIN mode: {mode}")


class InvertedVariateBlock(nn.Module):
    """Applies Multi-Head Self-Attention across the N=32 variate tokens."""

    def __init__(self, d_model: int = 256, n_heads: int = 8, d_ff: int = 512, dropout: float = 0.15) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # h shape: (batch, N_variates, d_model)
        h_norm = self.norm1(h)
        attn_out, attn_weights = self.attn(h_norm, h_norm, h_norm, need_weights=True)
        h = h + self.drop1(attn_out)
        h = h + self.ffn(self.norm2(h))
        return h, attn_weights


class MultivariateMesoITransformer(nn.Module):
    """
    32-Variate iTransformer for Tier 3 Meso Swing Prediction (15m/30m/1h/2h).
    Jointly outputs:
      1. 3-class Fee-Floored Triple-Barrier logits [down, flat, up] for BTCUSDT (variate 0).
      2. Multi-step quantile trajectory forecast (q10, q50, q90) over h=8 steps (2 hours).
      3. Interpretable 32x32 Cross-Variate Attention Matrix for regime diagnostics.
    """

    def __init__(
        self,
        seq_len: int = 192,          # 48 hours of 15m bars
        pred_len: int = 8,           # 2 hours ahead (8 x 15m)
        num_variates: int = 32,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if num_variates < 2:
            raise ValueError("MultivariateMesoITransformer requires num_variates >= 2 (got 1, which collapses variate attention!)")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_variates = num_variates

        self.revin = ReversibleInstanceNorm1d(num_variates=num_variates, affine=True)
        # Invert dimension: project each variate's temporal history (seq_len,) -> (d_model,)
        self.variate_projector = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.variate_id_embedding = nn.Embedding(num_variates, d_model)
        self.blocks = nn.ModuleList([
            InvertedVariateBlock(d_model=d_model, n_heads=n_heads, d_ff=d_model * 2, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # Head 1: BTCUSDT (variate 0) + Cross-Variate Global Context -> 3-class Triple-Barrier Classification
        self.cls_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3),
        )
        # Head 2: Quantile Regression Head (q10, q50, q90) over pred_len steps
        self.quantile_head = nn.Linear(d_model, pred_len * 3)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: Tensor of shape (batch, seq_len, num_variates=32)
        Returns:
            dict with keys: 'cls_logits' (B, 3), 'cls_probs' (B, 3),
            'quantiles' (B, pred_len, 3), 'variate_attention' (B, 32, 32)
        """
        bsz, seq_l, n_var = x.shape
        assert seq_l == self.seq_len and n_var == self.num_variates

        x_norm = self.revin(x, mode="norm")
        # Transpose to (batch, num_variates, seq_len) so each variate is a token!
        x_inv = x_norm.transpose(1, 2)

        var_ids = torch.arange(n_var, device=x.device).unsqueeze(0).expand(bsz, n_var)
        h = self.variate_projector(x_inv) + self.variate_id_embedding(var_ids)

        last_attn = None
        for block in self.blocks:
            h, last_attn = block(h)

        h = self.final_norm(h)
        btc_token = h[:, 0, :]                     # Variate 0 = BTCUSDT primary representation
        global_pool = h.mean(dim=1)                # Cross-market global representation
        combined = torch.cat([btc_token, global_pool], dim=-1)

        cls_logits = self.cls_head(combined)
        cls_probs = F.softmax(cls_logits, dim=-1)
        quantiles = self.quantile_head(btc_token).view(bsz, self.pred_len, 3)

        return {
            "cls_logits": cls_logits,
            "cls_probs": cls_probs,
            "quantiles": quantiles,
            "variate_attention": last_attn,
        }
