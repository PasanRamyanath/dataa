"""
feature engineering scripts/execution/hierarchical_sniper_engine.py
Production implementation of the 9-State Hierarchical Multi-Horizon Sniper
Execution Engine for Binance USD(S)-M Futures.

Key Innovations over legacy execution/maker_node.py & inference/binance_testnet_engine.py:
1. Decouples Swing Holding Horizon & Barrier Width (governed by Tier 3 15m-2h & Tier 2 1m-5m,
   where TP = 28..160 bps >> 4.0 bps Maker fee) from Entry/Exit Timing (governed by Tier 1 1s/10s).
2. Implements SNIPER_ENTRY_STALKING state: waits up to 180s for the 1s/10s microstructure
   ensemble to confirm dip exhaustion + safe FIFO queue fill probability before posting GTX limit.
3. Places resting Reduce-Only LIMIT Maker Take-Profit orders immediately upon entry fill to
   lock in FIFO queue priority and guarantee 2.0 bps Maker exit fees on TP hits.
4. Implements Microstructure Profit-Locking Radar & PASSIVE_MAKER_UNWIND (at 85% hold time)
   so expiring or stalling trades exit passively as a Maker at BBO instead of bleeding 5.0 bps
   Taker market-order fees + spread crossing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
import time
from typing import Literal

import numpy as np

from execution.maker_node import AdverseSelectionGuard, QueuePriorityTracker
from fusion.hierarchical_fusion_v3 import HierarchicalDecisionOutput

log = logging.getLogger("execution.hierarchical_sniper")

SniperState = Literal[
    "IDLE",
    "SNIPER_ENTRY_STALKING",
    "PLACING_MAKER_ENTRY",
    "RESTING_ENTRY_QUEUE",
    "PARTIAL_ENTRY_FILL",
    "FILLED_SWING_HOLDING",
    "PASSIVE_MAKER_UNWIND",
    "EMERGENCY_TAKER_EXIT",
    "CANCELING_AND_SYNC",
]


@dataclass
class ActiveSwingPosition:
    symbol: str
    side: Literal["BUY", "SELL"]          # BUY = Long, SELL = Short
    entry_price: float
    quantity: float
    entry_ts_ms: int
    target_holding_seconds: int
    take_profit_bps: float
    stop_loss_bps: float
    tp_price: float
    sl_price: float
    maker_entry_fee_bps: float = 2.0
    peak_favorable_excursion_bps: float = 0.0
    max_adverse_excursion_bps: float = 0.0
    trailing_maker_armed: bool = False
    resting_tp_order_id: int | None = None
    unwind_order_id: int | None = None


@dataclass
class ClosedSwingTradeRecord:
    symbol: str
    side: Literal["BUY", "SELL"]
    entry_price: float
    exit_price: float
    quantity: float
    entry_ts_ms: int
    exit_ts_ms: int
    holding_duration_sec: float
    gross_return_bps: float
    total_fee_bps: float
    net_return_bps: float
    peak_mfe_bps: float
    max_mae_bps: float
    exit_reason: Literal[
        "RESTING_MAKER_TP",
        "MICRO_TRAILING_MAKER_TP",
        "PASSIVE_MAKER_TIME_UNWIND",
        "MACRO_REVERSAL_MAKER_EXIT",
        "HARD_STOP_LOSS_TAKER",
        "HARD_DEADLINE_TAKER",
    ]


class HierarchicalSniperExecutionEngine:
    """
    9-State Hierarchical Multi-Horizon Sniper Execution Engine.
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        maker_fee_bps_per_leg: float = 2.0,
        taker_fee_bps_per_leg: float = 5.0,
        max_stalking_window_sec: int = 180,
        max_queue_resting_sec: int = 25,
        passive_unwind_start_frac: float = 0.85,
        trailing_activation_tp_frac: float = 0.55,
        trailing_giveback_frac: float = 0.30,
    ) -> None:
        self.symbol = symbol
        self.maker_fee_leg = maker_fee_bps_per_leg
        self.taker_fee_leg = taker_fee_bps_per_leg
        self.max_stalking_ms = max_stalking_window_sec * 1000
        self.max_queue_resting_ms = max_queue_resting_sec * 1000
        self.unwind_start_frac = passive_unwind_start_frac
        self.trailing_act_frac = trailing_activation_tp_frac
        self.trailing_giveback_frac = trailing_giveback_frac

        self.state: SniperState = "IDLE"
        self.stalking_decision: HierarchicalDecisionOutput | None = None
        self.stalking_start_ts_ms: int = 0

        self.resting_entry_side: Literal["BUY", "SELL"] | None = None
        self.resting_entry_price: float = 0.0
        self.resting_entry_qty: float = 0.0
        self.resting_entry_ts_ms: int = 0
        self.queue_tracker: QueuePriorityTracker | None = None
        self.adverse_guard: AdverseSelectionGuard | None = None

        self.active_position: ActiveSwingPosition | None = None
        self.closed_trades: list[ClosedSwingTradeRecord] = []

    def on_hierarchical_decision(
        self,
        decision: HierarchicalDecisionOutput,
        bbo_bid: float,
        bbo_ask: float,
        bbo_bid_qty: float,
        bbo_ask_qty: float,
        equity_usdt: float = 10000.0,
    ) -> str:
        """
        Processes a new 10s hierarchical decision across all 4 timeframe tiers.
        Returns the updated FSM state.
        """
        now_ms = decision.ts_ms
        if bbo_bid <= 0 or bbo_ask <= bbo_bid:
            return self.state

        # CASE 1: Engine is IDLE -> Check if Macro + Meso + Tactical authorize a new Swing Setup
        if self.state == "IDLE":
            if decision.execution_mode in ("PASSIVE_MAKER_SNIPER", "URGENT_HYBRID", "WAIT_FOR_MICRO_SNIPER"):
                if decision.primary_direction in (0, 2) and decision.expected_net_edge_bps >= 6.0:
                    self.stalking_decision = decision
                    self.stalking_start_ts_ms = now_ms
                    self.state = "SNIPER_ENTRY_STALKING"
                    # Immediately evaluate if current 1s/10s tick already triggers placement
                    if decision.execution_mode in ("PASSIVE_MAKER_SNIPER", "URGENT_HYBRID"):
                        self._place_sniper_maker_entry(
                            decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, now_ms
                        )
            return self.state

        # CASE 2: Currently STALKING an entry -> Check if 1s/10s sniper triggers or setup expired/invalidated
        if self.state == "SNIPER_ENTRY_STALKING":
            assert self.stalking_decision is not None
            # If higher-timeframe direction flipped against our stalking setup, abort immediately
            if decision.primary_direction != self.stalking_decision.primary_direction and decision.primary_direction != 1:
                self.stalking_decision = None
                self.state = "IDLE"
                return self.state

            # If stalking window (180s) expired without a clean 1s/10s entry, return to IDLE
            if (now_ms - self.stalking_start_ts_ms) > self.max_stalking_ms:
                self.stalking_decision = None
                self.state = "IDLE"
                return self.state

            # If 1s/10s Microstructure Sniper triggers now, place GTX BBO limit order!
            if decision.micro_sniper_trigger_prob >= 0.55:
                self._place_sniper_maker_entry(
                    self.stalking_decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, now_ms
                )
            return self.state

        # CASE 3: Holding an active Swing Position -> Evaluate Microstructure Profit-Locking & Macro Reversal
        if self.state in ("FILLED_SWING_HOLDING", "PASSIVE_MAKER_UNWIND"):
            self._manage_active_swing_position(decision, bbo_bid, bbo_ask, now_ms)
            return self.state

        return self.state

    def _place_sniper_maker_entry(
        self,
        decision: HierarchicalDecisionOutput,
        bbo_bid: float,
        bbo_ask: float,
        bbo_bid_qty: float,
        bbo_ask_qty: float,
        equity_usdt: float,
        now_ms: int,
    ) -> None:
        side: Literal["BUY", "SELL"] = "BUY" if decision.primary_direction == 2 else "SELL"
        limit_price = bbo_bid if side == "BUY" else bbo_ask
        supporting_qty = bbo_bid_qty if side == "BUY" else bbo_ask_qty

        notional = max(100.0, equity_usdt * max(0.10, decision.kelly_position_fraction))
        qty = round(notional / limit_price, 3)
        if qty <= 0:
            qty = 0.001

        self.resting_entry_side = side
        self.resting_entry_price = limit_price
        self.resting_entry_qty = qty
        self.resting_entry_ts_ms = now_ms
        self.queue_tracker = QueuePriorityTracker(initial_queue_ahead=supporting_qty)
        self.adverse_guard = AdverseSelectionGuard(
            initial_supporting_depth=supporting_qty,
            depth_deterioration_threshold=0.45,
            max_adverse_microprice_bps=1.8,
        )
        self.state = "RESTING_ENTRY_QUEUE"

    def on_market_micro_tick(
        self,
        ts_ms: int,
        bbo_bid: float,
        bbo_ask: float,
        bbo_bid_qty: float,
        bbo_ask_qty: float,
        aggressive_sell_vol_at_bid: float = 0.0,
        aggressive_buy_vol_at_ask: float = 0.0,
        eth_1s_ret_bps: float = 0.0,
    ) -> str:
        """
        Processes sub-second / 1s / 10s market updates for queue fill tracking,
        adverse selection cancellation, and barrier resolution.
        """
        mid = 0.5 * (bbo_bid + bbo_ask)

        # 1. Manage RESTING_ENTRY_QUEUE
        if self.state == "RESTING_ENTRY_QUEUE" and self.queue_tracker is not None and self.adverse_guard is not None:
            assert self.resting_entry_side is not None
            assert self.stalking_decision is not None

            # Check Adverse Selection Guard (L1/L2 depth collapse, microprice drift, or ETH 1s shock)
            supp_depth = bbo_bid_qty if self.resting_entry_side == "BUY" else bbo_ask_qty
            opp_depth = bbo_ask_qty if self.resting_entry_side == "BUY" else bbo_bid_qty
            microprice = (bbo_bid * opp_depth + bbo_ask * supp_depth) / max(supp_depth + opp_depth, 1e-8)

            if self.resting_entry_side == "BUY":
                adverse_drift_bps = ((self.resting_entry_price - microprice) / self.resting_entry_price) * 1e4
                cross_shock = eth_1s_ret_bps < -3.5
            else:
                adverse_drift_bps = ((microprice - self.resting_entry_price) / self.resting_entry_price) * 1e4
                cross_shock = eth_1s_ret_bps > 3.5

            if self.adverse_guard.should_cancel(self.resting_entry_side, supp_depth, adverse_drift_bps) or cross_shock:
                # Cancel resting order with 0.0 bps fee and return to SNIPER_ENTRY_STALKING!
                self.queue_tracker = None
                self.adverse_guard = None
                self.state = "SNIPER_ENTRY_STALKING"
                return self.state

            # Process aggressive volume executing against our resting BBO level
            if self.resting_entry_side == "BUY" and bbo_bid <= self.resting_entry_price:
                filled = self.queue_tracker.process_trade(aggressive_sell_vol_at_bid)
            elif self.resting_entry_side == "SELL" and bbo_ask >= self.resting_entry_price:
                filled = self.queue_tracker.process_trade(aggressive_buy_vol_at_ask)
            else:
                filled = False

            if filled:
                tp_bps = self.stalking_decision.take_profit_bps
                sl_bps = self.stalking_decision.stop_loss_bps
                ep = self.resting_entry_price
                if self.resting_entry_side == "BUY":
                    tp_p = ep * (1.0 + tp_bps * 1e-4)
                    sl_p = ep * (1.0 - sl_bps * 1e-4)
                else:
                    tp_p = ep * (1.0 - tp_bps * 1e-4)
                    sl_p = ep * (1.0 + sl_bps * 1e-4)

                self.active_position = ActiveSwingPosition(
                    symbol=self.symbol,
                    side=self.resting_entry_side,
                    entry_price=ep,
                    quantity=self.resting_entry_qty,
                    entry_ts_ms=ts_ms,
                    target_holding_seconds=self.stalking_decision.target_holding_seconds,
                    take_profit_bps=tp_bps,
                    stop_loss_bps=sl_bps,
                    tp_price=tp_p,
                    sl_price=sl_p,
                    maker_entry_fee_bps=self.maker_fee_leg,
                )
                self.state = "FILLED_SWING_HOLDING"
                return self.state

            # Re-peg / timeout if resting in queue > max_queue_resting_ms without filling
            if (ts_ms - self.resting_entry_ts_ms) > self.max_queue_resting_ms:
                self.queue_tracker = None
                self.adverse_guard = None
                self.state = "SNIPER_ENTRY_STALKING"
                return self.state

        # 2. Manage Active Position on Sub-Second / 1s Ticks
        if self.state in ("FILLED_SWING_HOLDING", "PASSIVE_MAKER_UNWIND") and self.active_position is not None:
            pos = self.active_position
            sign = 1.0 if pos.side == "BUY" else -1.0
            unrealized_bps = sign * ((mid - pos.entry_price) / pos.entry_price) * 1e4
            pos.peak_favorable_excursion_bps = max(pos.peak_favorable_excursion_bps, unrealized_bps)
            pos.max_adverse_excursion_bps = min(pos.max_adverse_excursion_bps, unrealized_bps)

            # A. Check Resting Maker Take-Profit Hit (2.0 bps entry + 2.0 bps Maker TP exit = 4.0 bps total fee!)
            if (pos.side == "BUY" and bbo_ask >= pos.tp_price) or (pos.side == "SELL" and bbo_bid <= pos.tp_price):
                self._close_position(
                    exit_price=pos.tp_price,
                    exit_ts_ms=ts_ms,
                    exit_fee_bps=self.maker_fee_leg,
                    reason="RESTING_MAKER_TP",
                )
                return self.state

            # B. Check Hard Stop-Loss Breach (2.0 bps Maker entry + 5.0 bps Taker SL exit = 7.0 bps total fee)
            if (pos.side == "BUY" and bbo_bid <= pos.sl_price) or (pos.side == "SELL" and bbo_ask >= pos.sl_price):
                exec_sl_p = bbo_bid if pos.side == "BUY" else bbo_ask
                self._close_position(
                    exit_price=exec_sl_p,
                    exit_ts_ms=ts_ms,
                    exit_fee_bps=self.taker_fee_leg,
                    reason="HARD_STOP_LOSS_TAKER",
                )
                return self.state

            # C. Check Trailing Maker Profit-Locking Activation
            if pos.peak_favorable_excursion_bps >= (self.trailing_act_frac * pos.take_profit_bps):
                pos.trailing_maker_armed = True
                drawback_from_peak = pos.peak_favorable_excursion_bps - unrealized_bps
                if drawback_from_peak >= (self.trailing_giveback_frac * pos.peak_favorable_excursion_bps) and unrealized_bps >= 8.0:
                    # Exit passively as Maker at BBO (Best Ask for Long, Best Bid for Short)!
                    maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
                    self._close_position(
                        exit_price=maker_exit_p,
                        exit_ts_ms=ts_ms,
                        exit_fee_bps=self.maker_fee_leg,
                        reason="MICRO_TRAILING_MAKER_TP",
                    )
                    return self.state

        return self.state

    def _manage_active_swing_position(
        self,
        decision: HierarchicalDecisionOutput,
        bbo_bid: float,
        bbo_ask: float,
        now_ms: int,
    ) -> None:
        if self.active_position is None:
            return
        pos = self.active_position
        elapsed_sec = (now_ms - pos.entry_ts_ms) / 1000.0
        hold_frac = elapsed_sec / max(1.0, float(pos.target_holding_seconds))

        sign = 1.0 if pos.side == "BUY" else -1.0
        d_micro = decision.micro_probs[2] - decision.micro_probs[0]
        d_meso = decision.meso_probs[2] - decision.meso_probs[0]

        # 1. If Meso + Tactical violently reverse against our position while in profit or small loss, exit as Maker!
        if (sign * d_meso) < -0.22 and (sign * d_micro) < -0.10:
            maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
            self._close_position(
                exit_price=maker_exit_p,
                exit_ts_ms=now_ms,
                exit_fee_bps=self.maker_fee_leg,
                reason="MACRO_REVERSAL_MAKER_EXIT",
            )
            return

        # 2. Enter PASSIVE_MAKER_UNWIND at 85% of target holding horizon
        if hold_frac >= self.unwind_start_frac and self.state == "FILLED_SWING_HOLDING":
            self.state = "PASSIVE_MAKER_UNWIND"

        if self.state == "PASSIVE_MAKER_UNWIND":
            # In unwind window (85% to 100% of horizon), wait for a 10s micro-bounce in our favor or micro exhaustion to exit as Maker!
            if (sign * d_micro) <= 0.05 or hold_frac >= 0.95:
                maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
                self._close_position(
                    exit_price=maker_exit_p,
                    exit_ts_ms=now_ms,
                    exit_fee_bps=self.maker_fee_leg,
                    reason="PASSIVE_MAKER_TIME_UNWIND",
                )
                return

        # 3. Hard 100% deadline safety fallback
        if hold_frac >= 1.05:
            taker_exit_p = bbo_bid if pos.side == "BUY" else bbo_ask
            self._close_position(
                exit_price=taker_exit_p,
                exit_ts_ms=now_ms,
                exit_fee_bps=self.taker_fee_leg,
                reason="HARD_DEADLINE_TAKER",
            )

    def _close_position(
        self,
        exit_price: float,
        exit_ts_ms: int,
        exit_fee_bps: float,
        reason: Literal[
            "RESTING_MAKER_TP",
            "MICRO_TRAILING_MAKER_TP",
            "PASSIVE_MAKER_TIME_UNWIND",
            "MACRO_REVERSAL_MAKER_EXIT",
            "HARD_STOP_LOSS_TAKER",
            "HARD_DEADLINE_TAKER",
        ],
    ) -> None:
        assert self.active_position is not None
        pos = self.active_position
        sign = 1.0 if pos.side == "BUY" else -1.0
        gross_bps = sign * ((exit_price - pos.entry_price) / pos.entry_price) * 1e4
        total_fee = pos.maker_entry_fee_bps + exit_fee_bps
        net_bps = gross_bps - total_fee
        dur_sec = max(0.0, (exit_ts_ms - pos.entry_ts_ms) / 1000.0)

        rec = ClosedSwingTradeRecord(
            symbol=pos.symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            entry_ts_ms=pos.entry_ts_ms,
            exit_ts_ms=exit_ts_ms,
            holding_duration_sec=round(dur_sec, 2),
            gross_return_bps=round(gross_bps, 4),
            total_fee_bps=round(total_fee, 4),
            net_return_bps=round(net_bps, 4),
            peak_mfe_bps=round(pos.peak_favorable_excursion_bps, 4),
            max_mae_bps=round(pos.max_adverse_excursion_bps, 4),
            exit_reason=reason,
        )
        self.closed_trades.append(rec)
        self.active_position = None
        self.stalking_decision = None
        self.state = "IDLE"
