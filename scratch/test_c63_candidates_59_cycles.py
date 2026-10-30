import os
import sys
import copy

sys.path.insert(0, os.path.abspath('feature engineering scripts'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from execution.champion_challenger_framework import (
    Cycle35_ReversionPureScalper,
    Cycle62_MidpointVetoScalper,
)
from simulate_all_59_cycles_qualification import run_full_simulation

archives = [d for d in os.listdir("scratch") if d.startswith("archive_cycle") and os.path.isdir(os.path.join("scratch", d))]
archives = sorted(archives, key=lambda x: int(x.replace("archive_cycle", "")))
valid = [d for d in archives if os.path.exists(os.path.join("scratch", d, "live_hierarchical_session_1h_ledger.jsonl")) and os.path.getsize(os.path.join("scratch", d, "live_hierarchical_session_1h_ledger.jsonl")) > 0]


class Candidate1_MidpointRetestPassiveScalper(Cycle62_MidpointVetoScalper):
    """
    Candidate 1: Combines Cycle 62 Winning Midpoint Veto with:
    1. Passive BBO Maker Unwind on Invalidation cuts (2.0 bps Maker fee instead of 5.0 bps Taker).
    2. Strict Post-Invalidation Retrace Guard (requires price to cross above/below last loss exit price before same-direction re-entry).
    """
    def __init__(self, symbol: str = "BTCUSDT"):
        super().__init__(symbol=symbol)
        self.strategy_profile = "Candidate 1: Midpoint Retest Passive Scalper"
        self.taker_fee_leg = 2.0  # Passive Maker Unwind fee at BBO
        self.last_loss_exit_price: float = 0.0
        self.last_loss_side: str = ""

    def _close_position(self, exit_price, exit_ts_ms, exit_fee_bps, reason="TAKE_PROFIT"):
        if self.active_position is not None:
            # If invalidation cut, use Maker fee at BBO
            if "INVALIDATION" in reason:
                exit_fee_bps = self.maker_fee_leg
                reason = "PASSIVE_MAKER_UNWIND"
            pos = self.active_position
            sign = 1.0 if pos.side == "BUY" else -1.0
            gross_bps = sign * ((exit_price - pos.entry_price) / pos.entry_price) * 1e4
            net_bps = gross_bps - (pos.maker_entry_fee_bps + exit_fee_bps)
            if net_bps <= 0:
                self.last_loss_exit_price = exit_price
                self.last_loss_side = pos.side
            else:
                self.last_loss_exit_price = 0.0
                self.last_loss_side = ""
        return super()._close_position(exit_price, exit_ts_ms, exit_fee_bps, reason)

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        if self.state == "IDLE":
            mid = 0.5 * (bbo_bid + bbo_ask)
            # Post-Invalidation Retrace Guard:
            if self.last_loss_exit_price > 0.0:
                if self.last_loss_side == "SELL" and decision.primary_direction == 0:
                    if mid <= self.last_loss_exit_price:
                        # Price hasn't recovered above the invalidation exit -> VETO same-side chop re-entry!
                        return self.state
                elif self.last_loss_side == "BUY" and decision.primary_direction == 2:
                    if mid >= self.last_loss_exit_price:
                        # Price hasn't dipped below the invalidation exit -> VETO same-side chop re-entry!
                        return self.state
        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)


class Candidate2_SteppedRatchetMidpointScalper(Candidate1_MidpointRetestPassiveScalper):
    """
    Candidate 2: Candidate 1 + Aggressive Stepped Ratchet
    Locks +5.0 bps gross (+1.0 bps net) once MFE reaches +8.0 bps.
    Locks +10.0 bps gross (+6.0 bps net) once MFE reaches +14.0 bps.
    """
    def __init__(self, symbol: str = "BTCUSDT"):
        super().__init__(symbol=symbol)
        self.strategy_profile = "Candidate 2: Stepped Ratchet Midpoint Scalper"
        self.min_trailing_activation_bps = 8.00
        self.tier1_min_lock_bps = 5.00


class Candidate3_HighEdgeExpansionRetestScalper(Candidate1_MidpointRetestPassiveScalper):
    """
    Candidate 3: Candidate 1 + High Edge Expansion (min_net_edge_bps >= 15.0 bps)
    Strictly filters out marginal noise trades.
    """
    def __init__(self, symbol: str = "BTCUSDT"):
        super().__init__(symbol=symbol)
        self.min_net_edge_bps = 15.0
        self.strategy_profile = "Candidate 3: High Edge Expansion Retest Scalper"


class Candidate4_MicroMomentumMidpointScalper(Candidate1_MidpointRetestPassiveScalper):
    """
    Candidate 4: Candidate 1 + Adverse 10s Micro-Momentum Veto (Cycle 60 100% WR solution)
    """
    def __init__(self, symbol: str = "BTCUSDT"):
        super().__init__(symbol=symbol)
        self.strategy_profile = "Candidate 4: Micro-Momentum Midpoint Scalper"
        self.adverse_micro_threshold = 0.08

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        if self.state == "IDLE":
            d_micro_up = decision.micro_probs[2] - decision.micro_probs[0]
            if decision.primary_direction == 0 and d_micro_up > self.adverse_micro_threshold:
                return self.state
            if decision.primary_direction == 2 and (-d_micro_up) > self.adverse_micro_threshold:
                return self.state
        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)


class Candidate5_AdaptiveRangeEnvelopeScalper(Candidate1_MidpointRetestPassiveScalper):
    """
    Candidate 5: Candidate 1 + Dynamic $6 Envelope around Range Extremes
    """
    def __init__(self, symbol: str = "BTCUSDT"):
        super().__init__(symbol=symbol)
        self.range_boundary_buffer_dollars = 6.0
        self.strategy_profile = "Candidate 5: Adaptive Range Envelope Scalper"


if __name__ == "__main__":
    print(f"RUNNING 59-CYCLE QUALIFICATION TOURNAMENT BATTERY ACROSS ALL CANDIDATES\n")
    candidates = [
        (lambda: Candidate1_MidpointRetestPassiveScalper(), "Candidate 1: Midpoint Retest Passive Scalper"),
        (lambda: Candidate2_SteppedRatchetMidpointScalper(), "Candidate 2: Stepped Ratchet Midpoint Scalper"),
        (lambda: Candidate3_HighEdgeExpansionRetestScalper(), "Candidate 3: High Edge Expansion Retest Scalper"),
        (lambda: Candidate4_MicroMomentumMidpointScalper(), "Candidate 4: Micro-Momentum Midpoint Scalper"),
        (lambda: Candidate5_AdaptiveRangeEnvelopeScalper(), "Candidate 5: Adaptive Range Envelope Scalper"),
    ]

    for factory, name in candidates:
        run_full_simulation(factory, name, valid)
