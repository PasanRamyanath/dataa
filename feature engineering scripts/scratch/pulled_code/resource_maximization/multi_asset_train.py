"""
Shared-backbone, per-asset-head multi-asset joint model (§13.2, plan-marked
OPTIONAL and explicitly "requires validation against the single-asset
baseline before being trusted" — that validation is the ablation harness in
this same package, resource_maximization/ablation_test.py, not this module).

Architecture, matching §13.2's description directly: one shared sequence
backbone (same block shape as models/ssm_challenger/model.SSMChallenger's
input_proj -> blocks -> final_norm, so a pretrained single-asset backbone's
weights can be loaded straight into this module's `backbone` if desired)
extracts a pooled hidden state from each asset's own window independently
(no cross-asset attention — the plan's version of "joint" here is
parameter-sharing, not information-sharing across assets at inference time,
which would leak future information from a correlated asset). Each asset
then gets its own small linear head, so BTC's decision boundary is never
forced to be identical to ETH's/SOL's/BNB's.

Only ever exercised in this repo against synthetic per-asset batches — no
real multi-asset checkpoint exists yet, matching this project's Kaggle-
training-excluded-for-now scope.

Usage:
    from resource_maximization.multi_asset_head import MultiAssetModel
    model = MultiAssetModel(n_features=len(FEATURE_COLUMNS), symbols=["BTCUSDT", *config.MULTI_ASSET_SYMBOLS])
    logits_by_symbol = model(x_by_symbol)   # {"BTCUSDT": (batch, 3), ...}

    python -m resource_maximization.multi_asset_head --demo   # synthetic data only
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn

import config
from models.ssm_challenger.model import _make_ssm_block

log = logging.getLogger("resource_maximization.multi_asset_head")


class MultiAssetModel(nn.Module):
    """Shared backbone + one independent direction head per symbol.
    `symbols` should list every asset this model serves, BTC included —
    the plan's multi-asset expansion adds ETHUSDT/SOLUSDT/BNBUSDT
    (config.MULTI_ASSET_SYMBOLS) alongside the existing BTC members, it
    doesn't replace them."""

    def __init__(self, n_features: int, symbols: list[str], d_model: int = 128,
                 n_layers: int = 4, n_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        if len(symbols) != len(set(symbols)):
            raise ValueError(f"duplicate symbols in {symbols!r}")
        self.symbols = list(symbols)
        self.input_proj = nn.Linear(n_features, d_model)
        self.blocks = nn.ModuleList([_make_ssm_block(d_model) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(d_model)
        self.heads = nn.ModuleDict({
            symbol: nn.Linear(d_model, n_classes) for symbol in self.symbols
        })

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Shared backbone only, (batch, seq, n_features) -> (batch, d_model)
        pooled hidden state. Exposed separately so callers (e.g. an
        uncertainty module) can reuse the same pooled representation without
        re-running a symbol-specific head."""
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        h = self.dropout(self.final_norm(h))
        return h[:, -1, :]

    def forward(self, x_by_symbol: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """x_by_symbol: {symbol: (batch, seq, n_features)}. Every batch is
        pushed through the SAME backbone weights (that's the parameter-
        sharing this module exists for) then routed to that symbol's own
        head. Symbols absent from x_by_symbol are simply skipped — a
        partial batch (e.g. one asset's feed lagging) never blocks the
        others."""
        out: dict[str, torch.Tensor] = {}
        for symbol, x in x_by_symbol.items():
            if symbol not in self.heads:
                raise KeyError(f"unknown symbol {symbol!r}, expected one of {self.symbols}")
            pooled = self.encode(x)
            out[symbol] = self.heads[symbol](pooled)
        return out

    def backbone_state_dict(self) -> dict:
        """Just the shared-backbone weights (input_proj/blocks/final_norm),
        so a single-asset SSMChallenger checkpoint's matching submodules can
        be copied in without dragging its direction_head/return_head along
        (those are single-class-count-specific and per-asset here anyway)."""
        return {
            k: v for k, v in self.state_dict().items()
            if k.startswith("input_proj.") or k.startswith("blocks.") or k.startswith("final_norm.")
        }


def _demo() -> None:
    torch.manual_seed(0)
    symbols = ["BTCUSDT", *config.MULTI_ASSET_SYMBOLS]
    n_features = 10
    model = MultiAssetModel(n_features=n_features, symbols=symbols, d_model=16, n_layers=1)
    model.eval()

    batch = {s: torch.randn(4, 20, n_features) for s in symbols}
    with torch.no_grad():
        out = model(batch)
    for symbol, logits in out.items():
        print(f"{symbol}: logits shape {tuple(logits.shape)}")

    partial = {"BTCUSDT": batch["BTCUSDT"]}
    with torch.no_grad():
        partial_out = model(partial)
    print("partial batch handled, symbols returned:", list(partial_out.keys()))

    shared_keys = list(model.backbone_state_dict().keys())
    print("shared backbone param groups:", len(shared_keys))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    if args.demo:
        _demo()
    else:
        parser.print_help()

