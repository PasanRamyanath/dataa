"""
Real-time Multi-Level Order Flow Imbalance (OFI) Calculator.
Processes top 20 Level-2 depth frames (@depth20@100ms) to compute 
microstructure imbalance across 6 percentage bands and decay metrics.
"""
from __future__ import annotations
import numpy as np

BOOK_DEPTH_LEVELS = [0.2, 1.0, 2.0, 3.0, 4.0, 5.0]
_LEVEL_MEAN = sum(BOOK_DEPTH_LEVELS) / len(BOOK_DEPTH_LEVELS)
_LEVEL_SS = sum((l - _LEVEL_MEAN) ** 2 for l in BOOK_DEPTH_LEVELS)


class MultiLevelOFICalculator:
    def __init__(self):
        self.levels = BOOK_DEPTH_LEVELS
        self.level_mean = _LEVEL_MEAN
        self.level_ss = _LEVEL_SS

    def compute_ofi_metrics(self, bids: list[list[str | float]], asks: list[list[str | float]]) -> dict[str, float]:
        """Computes multi-level OFI, near/wide OFI, decay rate, and decay slope from L2 depth snapshot.
        
        Args:
            bids: List of [price, qty] pairs descending by price.
            asks: List of [price, qty] pairs ascending by price.
            
        Returns:
            Dictionary containing all OFI feature metrics.
        """
        if not bids or not asks:
            return {
                "order_flow_imbalance_near": 0.0,
                "order_flow_imbalance_wide": 0.0,
                "ofi_level_0.2": 0.0,
                "ofi_level_1.0": 0.0,
                "ofi_level_2.0": 0.0,
                "ofi_level_3.0": 0.0,
                "ofi_level_4.0": 0.0,
                "ofi_level_5.0": 0.0,
                "ofi_imbalance_decay_rate": 0.0,
                "ofi_imbalance_decay_slope": 0.0,
            }

        try:
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            mid_price = (best_bid + best_ask) / 2.0
            if mid_price <= 0:
                mid_price = best_bid
        except Exception:
            mid_price = 1.0

        ofi_levels = {}
        for l in self.levels:
            pct_threshold = l / 100.0
            bid_cutoff = mid_price * (1.0 - pct_threshold)
            ask_cutoff = mid_price * (1.0 + pct_threshold)

            bid_qty_sum = sum(float(b[1]) for b in bids if float(b[0]) >= bid_cutoff)
            ask_qty_sum = sum(float(a[1]) for a in asks if float(a[0]) <= ask_cutoff)

            total_depth = bid_qty_sum + ask_qty_sum
            if total_depth > 1e-12:
                ofi_levels[l] = (bid_qty_sum - ask_qty_sum) / total_depth
            else:
                ofi_levels[l] = 0.0

        # Imbalance decay rate: dissipation from touch (0.2%) out to outer depth (5.0%)
        decay_rate = ofi_levels[0.2] - ofi_levels[5.0]

        # Linear regression slope: Sum((level_i - mean_level) * ofi_i) / Sum((level_i - mean_level)^2)
        slope_num = sum((l - self.level_mean) * ofi_levels[l] for l in self.levels)
        decay_slope = slope_num / self.level_ss if self.level_ss > 0 else 0.0

        # Near and wide summaries
        ofi_near = ofi_levels[1.0]  # <= 1% is near
        ofi_wide = ofi_levels[5.0]  # entire visible depth is wide

        return {
            "order_flow_imbalance_near": float(ofi_near),
            "order_flow_imbalance_wide": float(ofi_wide),
            "ofi_level_0.2": float(ofi_levels[0.2]),
            "ofi_level_1.0": float(ofi_levels[1.0]),
            "ofi_level_2.0": float(ofi_levels[2.0]),
            "ofi_level_3.0": float(ofi_levels[3.0]),
            "ofi_level_4.0": float(ofi_levels[4.0]),
            "ofi_level_5.0": float(ofi_levels[5.0]),
            "ofi_imbalance_decay_rate": float(decay_rate),
            "ofi_imbalance_decay_slope": float(decay_slope),
        }
