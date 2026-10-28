"""
feature engineering scripts/execution/champion_challenger_framework.py
Production Champion-Challenger Tournament Framework for Project-IMPJR Peak Trading System.

Cycle 30 Tournament Lineup:
  - 👑 Reigning Champion: WideBreatheTrailingScalper (Defended Cycle 29: produced +6.91 bps net cash gain on Trade 5)
  - ⚔️ Challenger 1: Patient180sStallBreathingScalper (Extends stall unwind from 90s to 180s, allowing 5m trends time to develop)
  - ⚔️ Challenger 2: ExtremeRangeBoundaryFilterScalper (Blocks SELL near session lows and BUY near session highs)
  - ⚔️ Challenger 3: AsymmetricWaveExpanderScalper (180s stall + 2.0x SL TP expansion on multi-horizon consensus)
  - ⚔️ Challenger 4: DualHorizonMomentumConfirmationScalper (Requires strict concordant tactical + meso alignment)
  - ⚔️ Challenger 5: DynamicHybridHighYieldPeakScalper (Combines 180s stall, range-boundary filter, wide-breathe trailing, and 0.12 threshold)
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
# 👑 REIGNING CHAMPION: WideBreatheTrailingScalper (Cycle 28 & 29 Champion)
# ==============================================================================
class WideBreatheTrailingScalper(HierarchicalSniperExecutionEngine):
    """
    Reigning Champion (Crowned in Cycle 28, Defended in Cycle 29):
    Combines:
    1. Wide-Breathe Trailing Shield: arms at >= 8.0 bps gross, locks min 5.5 bps gross (+1.5 bps net).
    2. Fast 90s Stalling Unwind: unwinds passive at BBO if MFE < 2.0 bps at 90s.
    3. Same-Side Retrace Price Barrier: requires $15.00 pullback before re-entering same side within 300s.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Reigning Champion: Wide-Breathe Trailing Scalper"
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
# ⚔️ CHALLENGER 1: Patient 180s Stall Breathing Scalper
# ==============================================================================
class Challenger1_Patient180sStallBreathingScalper(WideBreatheTrailingScalper):
    """
    Challenger 1: Patient 180s Stall Breathing Scalper
    Directly addresses the 90s stall bottleneck:
    Extends the stall unwind threshold from 90s to 180s (3 minutes) before initiating
    passive unwind, giving developing 5m Meso trends time to clear initial liquidity walls.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 1: Patient 180s Stall Breathing Scalper"

    def on_market_micro_tick(self, *args, **kwargs):
        state = super().on_market_micro_tick(*args, **kwargs)
        if self.active_position is not None:
            now_ms = kwargs.get("ts_ms", 0)
            elapsed_sec = (now_ms - self.active_position.entry_ts_ms) / 1000.0
            # 180s patient stall timer (instead of 90s)
            if elapsed_sec >= 180.0 and self.active_position.peak_favorable_excursion_bps < 2.0:
                pos = self.active_position
                bbo_bid = kwargs.get("bbo_bid", 0.0)
                bbo_ask = kwargs.get("bbo_ask", 0.0)
                exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
                self._close_position(
                    exit_price=exit_p,
                    exit_ts_ms=now_ms,
                    exit_fee_bps=self.maker_fee_leg,
                    reason="PATIENT_180S_DECAY_UNWIND",
                )
        return self.state


# ==============================================================================
# ⚔️ CHALLENGER 2: Extreme Range Boundary Filter Scalper
# ==============================================================================
class Challenger2_ExtremeRangeBoundaryFilterScalper(Challenger1_Patient180sStallBreathingScalper):
    """
    Challenger 2: Extreme Range Boundary Filter Scalper
    Tracks the rolling 15-minute price range.
    Blocks SELL orders when price is within $15.00 of the 15m session low (anti-bottom-selling).
    Blocks BUY orders when price is within $15.00 of the 15m session high (anti-top-buying).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 2: Extreme Range Boundary Filter Scalper"
        self.rolling_15m_prices: deque[tuple[int, float]] = deque(maxlen=90)

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        now_ms = decision.ts_ms
        mid = 0.5 * (bbo_bid + bbo_ask)
        self.rolling_15m_prices.append((now_ms, mid))

        if len(self.rolling_15m_prices) >= 18: # at least 3 minutes of history
            session_high = max(p for _, p in self.rolling_15m_prices)
            session_low = min(p for _, p in self.rolling_15m_prices)
            d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]

            # Range boundary filters
            if d_tact <= -0.14 and (mid <= session_low + 15.0):
                # Price is at the floor: do not sell into support
                return self.state
            elif d_tact >= 0.14 and (mid >= session_high - 15.0):
                # Price is at the ceiling: do not buy into resistance
                return self.state

        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)


# ==============================================================================
# ⚔️ CHALLENGER 3: Asymmetric Wave Expander Scalper (180s Stall + 2.0x SL TP)
# ==============================================================================
class Challenger3_AsymmetricWaveExpanderScalper(Challenger1_Patient180sStallBreathingScalper):
    """
    Challenger 3: Asymmetric Wave Expander Scalper
    Combines 180s patient stall timer with Take-Profit expansion:
    When Macro prob > 0.40 and Meso prob > 0.45, sets resting Maker TP to 2.0x SL (+29.0 bps).
    Uses Wide-Breathe Trailing (>= 8.0 bps) to protect banked gains.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 3: Asymmetric Wave Expander Scalper (2.0x TP)"

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
# ⚔️ CHALLENGER 4: Dual-Horizon Momentum Confirmation Scalper
# ==============================================================================
class Challenger4_DualHorizonMomentumConfirmationScalper(Challenger1_Patient180sStallBreathingScalper):
    """
    Challenger 4: Dual-Horizon Momentum Confirmation Scalper
    Requires concordant alignment between 1m Tactical AND 5m Meso before entering:
    - BUY: d_tact >= 0.14 AND d_meso >= 0.05
    - SELL: d_tact <= -0.14 AND d_meso <= -0.05
    Eliminates all single-timeframe noise and ensures trades ride multi-tier order flow.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 4: Dual-Horizon Momentum Confirmation Scalper"

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        now_ms = decision.ts_ms
        mid = 0.5 * (bbo_bid + bbo_ask)
        d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
        d_micro = decision.micro_probs[2] - decision.micro_probs[0]
        d_meso = decision.meso_probs[2] - decision.meso_probs[0]

        if self.state == "IDLE" and self.ticks_seen >= 10:
            tact_dir = 2 if d_tact >= 0.14 else (0 if d_tact <= -0.14 else 1)
            # Dual-horizon confirmation: tactical + meso must agree strongly
            tact_confirmed = (tact_dir in (0, 2)) and (
                (tact_dir == 2 and d_micro > 0.02 and d_meso >= 0.05) or
                (tact_dir == 0 and d_micro < -0.02 and d_meso <= -0.05)
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
# ⚔️ CHALLENGER 5: Dynamic Hybrid High-Yield Peak Scalper
# ==============================================================================
class Challenger5_DynamicHybridHighYieldPeakScalper(Challenger2_ExtremeRangeBoundaryFilterScalper):
    """
    Challenger 5: Dynamic Hybrid High-Yield Peak Scalper
    Integrates all four peak discoveries:
    1. Patient 180s stall timer
    2. Extreme range-boundary support/resistance filter
    3. Wide-Breathe Trailing (>= 8.0 bps activation, 5.5 bps min lock)
    4. Calibrated 0.12 tactical impulse threshold for 4-6 trades/hr velocity
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 5: Dynamic Hybrid High-Yield Peak Scalper"

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
                (tact_dir == 2 and d_micro > 0.015 and d_meso >= 0.00) or
                (tact_dir == 0 and d_micro < -0.015 and d_meso <= 0.00)
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
                strategy_profile="Reigning Champion: Wide-Breathe Trailing Scalper",
            ),
            "Challenger_1": TournamentCompetitorRecord(
                name="Challenger_1",
                engine=Challenger1_Patient180sStallBreathingScalper(symbol=symbol),
                strategy_profile="Patient 180s Stall Breathing Scalper",
            ),
            "Challenger_2": TournamentCompetitorRecord(
                name="Challenger_2",
                engine=Challenger2_ExtremeRangeBoundaryFilterScalper(symbol=symbol),
                strategy_profile="Extreme Range Boundary Filter Scalper",
            ),
            "Challenger_3": TournamentCompetitorRecord(
                name="Challenger_3",
                engine=Challenger3_AsymmetricWaveExpanderScalper(symbol=symbol),
                strategy_profile="Asymmetric Wave Expander Scalper (2.0x TP)",
            ),
            "Challenger_4": TournamentCompetitorRecord(
                name="Challenger_4",
                engine=Challenger4_DualHorizonMomentumConfirmationScalper(symbol=symbol),
                strategy_profile="Dual-Horizon Momentum Confirmation Scalper",
            ),
            "Challenger_5": TournamentCompetitorRecord(
                name="Challenger_5",
                engine=Challenger5_DynamicHybridHighYieldPeakScalper(symbol=symbol),
                strategy_profile="Dynamic Hybrid High-Yield Peak Scalper",
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
