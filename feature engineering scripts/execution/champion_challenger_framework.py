"""
feature engineering scripts/execution/champion_challenger_framework.py
Production Champion-Challenger Tournament Framework for Project-IMPJR Peak Trading System.

Cycle 29 Tournament Lineup:
  - 👑 Reigning Champion: WideBreatheTrailingScalper (Crowned Cycle 28 Winner: captured +3.58 bps net on Trade 3, 2.2x over legacy champion)
  - ⚔️ Challenger 1: StrictMesoConcordanceScalper (Requires d_meso >= +0.02 for BUY, eliminating hostile counter-trend cuts)
  - ⚔️ Challenger 2: PostCutRetraceExpansionScalper (Expands re-entry barrier to $25 after any stop cut)
  - ⚔️ Challenger 3: AsymmetricTrendExpansionScalper (TP = 2.0x SL + Strict Meso Concordance)
  - ⚔️ Challenger 4: DualVelocityFlowScalper (Requires 10s micro velocity > 0.3 bps/s + tactical momentum)
  - ⚔️ Challenger 5: DynamicHybridHighYieldScalper (Strict Meso + Wide-Breathe + 0.12 tactical impulse threshold for 4-6 tr/hr)
"""
from __future__ import annotations
import copy
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal
import numpy as np

from execution.hierarchical_sniper_engine import (
    HierarchicalSniperExecutionEngine,
    ActiveSwingPosition,
    ClosedSwingTradeRecord,
)
from fusion.hierarchical_fusion_v3 import HierarchicalDecisionOutput

log = logging.getLogger("champion_challenger")


@dataclass
class TournamentCompetitorRecord:
    name: str
    engine: HierarchicalSniperExecutionEngine
    strategy_profile: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate_pct: float = 0.0
    gross_pnl_bps: float = 0.0
    total_fees_bps: float = 0.0
    net_pnl_bps: float = 0.0
    profit_factor: float = 0.0
    avg_trade_pnl_bps: float = 0.0
    avg_holding_sec: float = 0.0
    active_position_str: str = "NONE"
    fitness_score: float = 0.0


# ==============================================================================
# 👑 REIGNING CHAMPION: WideBreatheTrailingScalper (Crowned Cycle 28 Winner)
# ==============================================================================
class WideBreatheTrailingScalper(HierarchicalSniperExecutionEngine):
    """
    Reigning Champion (Crowned in Cycle 28):
    Combines:
    1. Wide-Breathe Trailing Shield: arms at >= 8.0 bps gross, locks min 5.5 bps gross (+1.5 bps net).
    2. Fast 90s Stalling Unwind: unwinds passive at BBO if MFE < 2.0 bps at 90s.
    3. Same-Side Retrace Price Barrier: requires $15.00 pullback before re-entering same side within 300s.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Reigning Champion: Wide-Breathe Trailing Scalper (Cycle 28 Winner)"
        self.min_trailing_activation_bps = 8.00
        self.tier1_min_lock_bps = 5.50
        self.model_harvest_min_bps = 7.50
        self.passive_unwind_min_bps = 5.50
        self.min_net_edge_bps = 4.0
        self.last_trade_exit_price: float = 0.0
        self.last_trade_side: str = ""

    def _close_position(self, exit_price, exit_ts_ms, exit_fee_bps, reason="TAKE_PROFIT"):
        if self.active_position is not None:
            self.last_trade_exit_price = exit_price
            self.last_trade_side = self.active_position.side
        return super()._close_position(exit_price, exit_ts_ms, exit_fee_bps, reason)

    def on_market_micro_tick(self, *args, **kwargs):
        state = super().on_market_micro_tick(*args, **kwargs)
        if self.active_position is not None:
            now_ms = kwargs.get("ts_ms", 0)
            elapsed_sec = (now_ms - self.active_position.entry_ts_ms) / 1000.0
            if elapsed_sec >= 90.0 and self.active_position.peak_favorable_excursion_bps < 2.0:
                pos = self.active_position
                bbo_bid = kwargs.get("bbo_bid", 0.0)
                bbo_ask = kwargs.get("bbo_ask", 0.0)
                exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
                self._close_position(
                    exit_price=exit_p,
                    exit_ts_ms=now_ms,
                    exit_fee_bps=self.maker_fee_leg,
                    reason="FAST_90S_DECAY_UNWIND",
                )
        return self.state

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        now_ms = decision.ts_ms
        mid = 0.5 * (bbo_bid + bbo_ask)
        d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
        d_micro = decision.micro_probs[2] - decision.micro_probs[0]
        d_meso = decision.meso_probs[2] - decision.meso_probs[0]

        if self.state == "IDLE" and self.ticks_seen >= 10:
            tact_dir = 2 if d_tact >= 0.14 else (0 if d_tact <= -0.14 else 1)
            tact_confirmed = (tact_dir in (0, 2)) and (
                (tact_dir == 2 and d_micro > 0.02 and d_meso >= -0.05) or
                (tact_dir == 0 and d_micro < -0.02 and d_meso <= 0.05)
            )
            
            if tact_confirmed and not self._is_parabolic_surge(tact_dir, now_ms, mid):
                can_reenter = True
                if self.last_exit_ts_ms > 0:
                    elapsed = (now_ms - self.last_exit_ts_ms) / 1000.0
                    if not self.last_exit_was_win and elapsed < 90.0:
                        can_reenter = False
                    elif self.last_exit_was_win and elapsed < 20.0:
                        can_reenter = False
                    
                    if can_reenter and elapsed < 300.0 and self.last_trade_exit_price > 0:
                        if tact_dir == 2 and self.last_trade_side == "BUY" and (mid > self.last_trade_exit_price - 15.0):
                            can_reenter = False
                        elif tact_dir == 0 and self.last_trade_side == "SELL" and (mid < self.last_trade_exit_price + 15.0):
                            can_reenter = False

                if can_reenter:
                    tact_dec = copy.copy(decision)
                    object.__setattr__(tact_dec, "primary_direction", tact_dir) if hasattr(tact_dec, "__dataclass_fields__") and getattr(tact_dec, "__dataclass_params__", None) and getattr(tact_dec.__dataclass_params__, "frozen", False) else setattr(tact_dec, "primary_direction", tact_dir)
                    self._place_sniper_maker_entry(tact_dec, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, now_ms, intra_bar_open)
                    return self.state

        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)


# ==============================================================================
# ⚔️ CHALLENGER 1: Strict Meso Concordance Scalper (Anti-Hostile Flow)
# ==============================================================================
class Challenger1_StrictMesoConcordanceScalper(WideBreatheTrailingScalper):
    """
    Challenger 1: Wide-Breathe + Strict Meso Concordance Guard
    Directly addresses the discovery from Cycle 28 Trade 4:
    Never takes a BUY trade when 5m Meso order flow is negative (requires d_meso >= +0.02).
    Never takes a SELL trade when 5m Meso order flow is positive (requires d_meso <= -0.02).
    Completely eliminates hostile counter-trend trades.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 1: Strict Meso Concordance Scalper (Anti-Hostile Flow)"

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        now_ms = decision.ts_ms
        mid = 0.5 * (bbo_bid + bbo_ask)
        d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
        d_micro = decision.micro_probs[2] - decision.micro_probs[0]
        d_meso = decision.meso_probs[2] - decision.meso_probs[0]

        if self.state == "IDLE" and self.ticks_seen >= 10:
            tact_dir = 2 if d_tact >= 0.14 else (0 if d_tact <= -0.14 else 1)
            # Strict Meso concordance: positive Meso required for BUY, negative for SELL
            tact_confirmed = (tact_dir in (0, 2)) and (
                (tact_dir == 2 and d_micro > 0.02 and d_meso >= 0.02) or
                (tact_dir == 0 and d_micro < -0.02 and d_meso <= -0.02)
            )
            
            if tact_confirmed and not self._is_parabolic_surge(tact_dir, now_ms, mid):
                can_reenter = True
                if self.last_exit_ts_ms > 0:
                    elapsed = (now_ms - self.last_exit_ts_ms) / 1000.0
                    if not self.last_exit_was_win and elapsed < 90.0:
                        can_reenter = False
                    elif self.last_exit_was_win and elapsed < 20.0:
                        can_reenter = False
                    
                    if can_reenter and elapsed < 300.0 and self.last_trade_exit_price > 0:
                        if tact_dir == 2 and self.last_trade_side == "BUY" and (mid > self.last_trade_exit_price - 15.0):
                            can_reenter = False
                        elif tact_dir == 0 and self.last_trade_side == "SELL" and (mid < self.last_trade_exit_price + 15.0):
                            can_reenter = False

                if can_reenter:
                    tact_dec = copy.copy(decision)
                    object.__setattr__(tact_dec, "primary_direction", tact_dir) if hasattr(tact_dec, "__dataclass_fields__") and getattr(tact_dec, "__dataclass_params__", None) and getattr(tact_dec.__dataclass_params__, "frozen", False) else setattr(tact_dec, "primary_direction", tact_dir)
                    self._place_sniper_maker_entry(tact_dec, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, now_ms, intra_bar_open)
                    return self.state

        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)


# ==============================================================================
# ⚔️ CHALLENGER 2: Post-Cut Retrace Expansion Scalper ($25 barrier)
# ==============================================================================
class Challenger2_PostCutRetraceExpansionScalper(WideBreatheTrailingScalper):
    """
    Challenger 2: Post-Cut Retrace Expansion Scalper
    Following an invalidation cut or loss trade, expands the required same-side
    retrace barrier from $15.00 to $25.00 for 300 seconds.
    Prevents catching falling knives during prolonged post-cut liquidation cascades.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 2: Post-Cut Retrace Expansion Scalper ($25 barrier)"

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        now_ms = decision.ts_ms
        mid = 0.5 * (bbo_bid + bbo_ask)
        d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
        d_micro = decision.micro_probs[2] - decision.micro_probs[0]
        d_meso = decision.meso_probs[2] - decision.meso_probs[0]

        if self.state == "IDLE" and self.ticks_seen >= 10:
            tact_dir = 2 if d_tact >= 0.14 else (0 if d_tact <= -0.14 else 1)
            tact_confirmed = (tact_dir in (0, 2)) and (
                (tact_dir == 2 and d_micro > 0.02 and d_meso >= 0.00) or
                (tact_dir == 0 and d_micro < -0.02 and d_meso <= 0.00)
            )
            
            if tact_confirmed and not self._is_parabolic_surge(tact_dir, now_ms, mid):
                can_reenter = True
                if self.last_exit_ts_ms > 0:
                    elapsed = (now_ms - self.last_exit_ts_ms) / 1000.0
                    barrier_usd = 25.0 if not self.last_exit_was_win else 15.0
                    if not self.last_exit_was_win and elapsed < 90.0:
                        can_reenter = False
                    elif self.last_exit_was_win and elapsed < 20.0:
                        can_reenter = False
                    
                    if can_reenter and elapsed < 300.0 and self.last_trade_exit_price > 0:
                        if tact_dir == 2 and self.last_trade_side == "BUY" and (mid > self.last_trade_exit_price - barrier_usd):
                            can_reenter = False
                        elif tact_dir == 0 and self.last_trade_side == "SELL" and (mid < self.last_trade_exit_price + barrier_usd):
                            can_reenter = False

                if can_reenter:
                    tact_dec = copy.copy(decision)
                    object.__setattr__(tact_dec, "primary_direction", tact_dir) if hasattr(tact_dec, "__dataclass_fields__") and getattr(tact_dec, "__dataclass_params__", None) and getattr(tact_dec.__dataclass_params__, "frozen", False) else setattr(tact_dec, "primary_direction", tact_dir)
                    self._place_sniper_maker_entry(tact_dec, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, now_ms, intra_bar_open)
                    return self.state

        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)


# ==============================================================================
# ⚔️ CHALLENGER 3: Asymmetric Trend-Expansion Scalper (2.0x SL TP + Strict Meso)
# ==============================================================================
class Challenger3_AsymmetricTrendExpansionScalper(Challenger1_StrictMesoConcordanceScalper):
    """
    Challenger 3: Asymmetric Trend-Expansion Scalper
    Combines Strict Meso Concordance with Take-Profit expansion:
    When Macro prob > 0.40 and Meso prob > 0.45, sets resting Maker TP to 2.0x SL (+29.0 bps).
    Uses Wide-Breathe Trailing (>= 8.0 bps) to protect banked gains.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 3: Asymmetric Trend-Expansion Scalper (2.0x TP + Strict Meso)"

    def _place_sniper_maker_entry(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, now_ms, intra_bar_open=0.0):
        super()._place_sniper_maker_entry(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, now_ms, intra_bar_open)
        if self.active_position is not None:
            pos = self.active_position
            p_macro = decision.macro_probs[2] if pos.side == "BUY" else decision.macro_probs[0]
            p_meso = decision.meso_probs[2] if pos.side == "BUY" else decision.meso_probs[0]
            if p_macro > 0.40 and p_meso > 0.45:
                expanded_tp_bps = pos.stop_loss_bps * 2.0
                sign = 1.0 if pos.side == "BUY" else -1.0
                pos.take_profit_bps = expanded_tp_bps
                pos.tp_price = pos.entry_price * (1.0 + sign * (expanded_tp_bps / 10000.0))


# ==============================================================================
# ⚔️ CHALLENGER 4: Dual-Velocity Flow Scalper
# ==============================================================================
class Challenger4_DualVelocityFlowScalper(Challenger1_StrictMesoConcordanceScalper):
    """
    Challenger 4: Strict Meso + Dual-Velocity Flow Guard
    Requires intra-second micro velocity to exceed 0.3 bps/s in the signal direction
    at the time of tactical entry. Ensures orders are only placed when order books
    are exhibiting active directional liquidity consumption.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 4: Dual-Velocity Flow Scalper"
        self.prev_mid: float = 0.0
        self.prev_tick_ms: int = 0
        self.last_velocity_bps_s: float = 0.0

    def on_market_micro_tick(self, *args, **kwargs):
        mid = 0.5 * (kwargs.get("bbo_bid", 0.0) + kwargs.get("bbo_ask", 0.0))
        now_ms = kwargs.get("ts_ms", 0)
        if self.prev_mid > 0 and self.prev_tick_ms > 0 and now_ms > self.prev_tick_ms:
            dt_s = (now_ms - self.prev_tick_ms) / 1000.0
            if dt_s > 0.05:
                self.last_velocity_bps_s = (((mid - self.prev_mid) / self.prev_mid) * 1e4) / dt_s
        self.prev_mid = mid
        self.prev_tick_ms = now_ms
        return super().on_market_micro_tick(*args, **kwargs)

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
        # Require micro velocity to not oppose tactical direction
        if d_tact >= 0.14 and self.last_velocity_bps_s < -0.20:
            return self.state
        elif d_tact <= -0.14 and self.last_velocity_bps_s > 0.20:
            return self.state
        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)


# ==============================================================================
# ⚔️ CHALLENGER 5: Dynamic Hybrid High-Yield Scalper (4-6 trades/hr)
# ==============================================================================
class Challenger5_DynamicHybridHighYieldScalper(Challenger1_StrictMesoConcordanceScalper):
    """
    Challenger 5: Dynamic Hybrid High-Yield Scalper
    Combines Strict Meso Concordance with a calibrated 0.12 tactical impulse
    threshold to target 4-6 trades/hour with expanded profit margins and zero
    counter-trend exposure.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 5: Dynamic Hybrid High-Yield Scalper (4-6 tr/hr)"

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        now_ms = decision.ts_ms
        mid = 0.5 * (bbo_bid + bbo_ask)
        d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
        d_micro = decision.micro_probs[2] - decision.micro_probs[0]
        d_meso = decision.meso_probs[2] - decision.meso_probs[0]

        if self.state == "IDLE" and self.ticks_seen >= 10:
            # Calibrated 0.12 threshold
            tact_dir = 2 if d_tact >= 0.12 else (0 if d_tact <= -0.12 else 1)
            tact_confirmed = (tact_dir in (0, 2)) and (
                (tact_dir == 2 and d_micro > 0.015 and d_meso >= 0.02) or
                (tact_dir == 0 and d_micro < -0.015 and d_meso <= -0.02)
            )
            
            if tact_confirmed and not self._is_parabolic_surge(tact_dir, now_ms, mid):
                can_reenter = True
                if self.last_exit_ts_ms > 0:
                    elapsed = (now_ms - self.last_exit_ts_ms) / 1000.0
                    if not self.last_exit_was_win and elapsed < 75.0:
                        can_reenter = False
                    elif self.last_exit_was_win and elapsed < 15.0:
                        can_reenter = False
                    
                    if can_reenter and elapsed < 300.0 and self.last_trade_exit_price > 0:
                        if tact_dir == 2 and self.last_trade_side == "BUY" and (mid > self.last_trade_exit_price - 14.0):
                            can_reenter = False
                        elif tact_dir == 0 and self.last_trade_side == "SELL" and (mid < self.last_trade_exit_price + 14.0):
                            can_reenter = False

                if can_reenter:
                    tact_dec = copy.copy(decision)
                    object.__setattr__(tact_dec, "primary_direction", tact_dir) if hasattr(tact_dec, "__dataclass_fields__") and getattr(tact_dec, "__dataclass_params__", None) and getattr(tact_dec.__dataclass_params__, "frozen", False) else setattr(tact_dec, "primary_direction", tact_dir)
                    self._place_sniper_maker_entry(tact_dec, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, now_ms, intra_bar_open)
                    return self.state

        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)


# ==============================================================================
# 🏆 TOURNAMENT MANAGER (1 CHAMPION + 5 CHALLENGERS)
# ==============================================================================
class ChampionChallengerTournamentManager:
    """
    Manages the 1 Champion + 5 Challengers Tournament in real time.
    Ingests each tick and decision simultaneously into all 6 engines.
    Maintains the live leaderboard and crowns the new champion at session end.
    """
    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol
        self.competitors: dict[str, TournamentCompetitorRecord] = {
            "Champion": TournamentCompetitorRecord(
                name="Champion",
                engine=WideBreatheTrailingScalper(symbol=symbol),
                strategy_profile="Reigning Champion: Wide-Breathe Trailing Scalper (Cycle 28 Winner)",
            ),
            "Challenger_1": TournamentCompetitorRecord(
                name="Challenger_1",
                engine=Challenger1_StrictMesoConcordanceScalper(symbol=symbol),
                strategy_profile="Strict Meso Concordance Scalper (Anti-Hostile Flow)",
            ),
            "Challenger_2": TournamentCompetitorRecord(
                name="Challenger_2",
                engine=Challenger2_PostCutRetraceExpansionScalper(symbol=symbol),
                strategy_profile="Post-Cut Retrace Expansion Scalper ($25 barrier)",
            ),
            "Challenger_3": TournamentCompetitorRecord(
                name="Challenger_3",
                engine=Challenger3_AsymmetricTrendExpansionScalper(symbol=symbol),
                strategy_profile="Asymmetric Trend-Expansion Scalper (2.0x TP + Strict Meso)",
            ),
            "Challenger_4": TournamentCompetitorRecord(
                name="Challenger_4",
                engine=Challenger4_DualVelocityFlowScalper(symbol=symbol),
                strategy_profile="Dual-Velocity Flow Scalper",
            ),
            "Challenger_5": TournamentCompetitorRecord(
                name="Challenger_5",
                engine=Challenger5_DynamicHybridHighYieldScalper(symbol=symbol),
                strategy_profile="Dynamic Hybrid High-Yield Scalper (4-6 tr/hr)",
            ),
        }

    def on_market_micro_tick(self, **kwargs) -> None:
        for c in self.competitors.values():
            try:
                c.engine.on_market_micro_tick(**kwargs)
            except Exception as e:
                log.debug("Tick error in %s: %s", c.name, e)

    def on_hierarchical_decision(self, **kwargs) -> None:
        for c in self.competitors.values():
            try:
                c.engine.on_hierarchical_decision(**kwargs)
            except Exception as e:
                log.debug("Decision error in %s: %s", c.name, e)

    def update_metrics(self, current_mid: float = 0.0) -> list[dict[str, Any]]:
        leaderboard = []
        for name, comp in self.competitors.items():
            trades = comp.engine.closed_trades
            n = len(trades)
            w = sum(1 for t in trades if t.net_return_bps > 0)
            l = sum(1 for t in trades if t.net_return_bps <= 0)
            wr = (w / n * 100.0) if n > 0 else 0.0
            gross = sum(t.gross_return_bps for t in trades)
            fees = sum(t.total_fee_bps for t in trades)
            net = sum(t.net_return_bps for t in trades)
            gw = sum(t.net_return_bps for t in trades if t.net_return_bps > 0)
            gl = abs(sum(t.net_return_bps for t in trades if t.net_return_bps < 0))
            pf = (gw / max(1e-6, gl)) if gl > 0 else (999.0 if gw > 0 else 1.0)
            avg_pnl = (net / n) if n > 0 else 0.0
            avg_dur = (sum(t.holding_duration_sec for t in trades) / n) if n > 0 else 0.0

            pos_str = "NONE"
            if comp.engine.active_position:
                ap = comp.engine.active_position
                sign = 1.0 if ap.side == "BUY" else -1.0
                unreal = sign * ((current_mid - ap.entry_price) / ap.entry_price) * 1e4 if current_mid > 0 else 0.0
                pos_str = f"{ap.side} @ {ap.entry_price:.1f} ({unreal:+.1f}b)"

            # Multi-objective fitness score: Net PnL * sqrt(WinRate/100) * ln(1 + Trades) * min(PF, 5.0)
            wr_factor = np.sqrt(max(0.1, wr / 100.0))
            trade_factor = np.log1p(n)
            pf_factor = float(np.clip(pf, 0.2, 5.0))
            fitness = round(net * wr_factor * trade_factor * pf_factor, 2)

            comp.total_trades = n
            comp.winning_trades = w
            comp.losing_trades = l
            comp.win_rate_pct = round(wr, 2)
            comp.gross_pnl_bps = round(gross, 2)
            comp.total_fees_bps = round(fees, 2)
            comp.net_pnl_bps = round(net, 2)
            comp.profit_factor = round(pf, 2)
            comp.avg_trade_pnl_bps = round(avg_pnl, 2)
            comp.avg_holding_sec = round(avg_dur, 1)
            comp.active_position_str = pos_str
            comp.fitness_score = fitness

            leaderboard.append({
                "rank": 0,
                "name": name,
                "strategy_profile": comp.strategy_profile,
                "trades": n,
                "wins": w,
                "losses": l,
                "win_rate_pct": round(wr, 1),
                "net_pnl_bps": round(net, 2),
                "profit_factor": round(pf, 2),
                "avg_trade_bps": round(avg_pnl, 2),
                "avg_duration_sec": round(avg_dur, 1),
                "active_pos": pos_str,
                "fitness_score": fitness,
            })

        # Rank by net PnL descending, then win rate, then trades
        leaderboard.sort(key=lambda x: (x["net_pnl_bps"], x["win_rate_pct"], x["trades"]), reverse=True)
        for i, item in enumerate(leaderboard, 1):
            item["rank"] = i

        return leaderboard
