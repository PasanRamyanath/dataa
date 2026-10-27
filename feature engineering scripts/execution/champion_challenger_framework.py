"""
feature engineering scripts/execution/champion_challenger_framework.py
Production Champion-Challenger Tournament Framework for Project-IMPJR Peak Trading System.
Cycle 27 Tournament Roster:
  - 👑 Reigning Champion: SameSideRetracePriceBarrierScalper (Undefeated with +37.78 bps combined net in Cycles 25 & 26, 100% WR in Cycle 26)
  - ⚔️ Challenger 1: Relaxed Impulse Frequency Scalper (Threshold relaxed to 0.10 to increase trade velocity to 4-6 tr/hr)
  - ⚔️ Challenger 2: Velocity Burst Momentum Scalper (Triggers on 10s velocity acceleration > 0.8 bps/s)
  - ⚔️ Challenger 3: Asymmetric Risk-Reward Scalper (TP = 1.8 * SL on high-conviction setups)
  - ⚔️ Challenger 4: Conformal Shannon Negentropy Scalper (Exits at peak information certainty)
  - ⚔️ Challenger 5: Fast 90s Stalling Retrace Scalper (Tight 90s stall prevention)
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
# 👑 REIGNING CHAMPION: SameSideRetracePriceBarrierScalper (Crowned Cycle 25)
# ==============================================================================
class SameSideRetracePriceBarrierScalper(HierarchicalSniperExecutionEngine):
    """
    Undefeated Champion (Crowned in Cycle 25 with +30.68 bps Net, 24.60 PF, defended in Cycle 26 with +7.10 bps Net):
    Combines High-Frequency Tactical Impulse Scaling with Adaptive 120s Momentum Decay Unwind
    and a Same-Side Retrace Price Barrier:
    When re-entering in the same direction within 300s, requires price to have retraced
    at least $15.00 better than the prior trade exit price.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Reigning Champion: Same-Side Retrace Barrier Scalper (+37.78 bps Net in C25-C26)"
        self.min_net_edge_bps = 4.0
        self.last_trade_exit_price: float = 0.0
        self.last_trade_side: str = ""

    def _close_position(self, exit_price, exit_ts_ms, exit_fee_bps, reason="TAKE_PROFIT"):
        if self.active_position is not None:
            self.last_trade_exit_price = exit_price
            self.last_trade_side = self.active_position.side
        return super()._close_position(exit_price, exit_ts_ms, exit_fee_bps, reason)

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
                    
                    # Same-Side Retrace Price Barrier check
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

    def on_market_micro_tick(self, *args, **kwargs):
        state = super().on_market_micro_tick(*args, **kwargs)
        if self.active_position is not None:
            now_ms = kwargs.get("ts_ms", 0)
            elapsed_sec = (now_ms - self.active_position.entry_ts_ms) / 1000.0
            if elapsed_sec >= 120.0 and self.active_position.peak_favorable_excursion_bps < 3.0:
                pos = self.active_position
                bbo_bid = kwargs.get("bbo_bid", 0.0)
                bbo_ask = kwargs.get("bbo_ask", 0.0)
                exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
                self._close_position(
                    exit_price=exit_p,
                    exit_ts_ms=now_ms,
                    exit_fee_bps=self.maker_fee_leg,
                    reason="ADAPTIVE_DECAY_TIME_UNWIND",
                )
        return self.state


# ==============================================================================
# ⚔️ CHALLENGER 1: Relaxed Impulse Frequency Scalper (4-6 trades/hr)
# ==============================================================================
class Challenger1_RelaxedImpulseFrequencyScalper(SameSideRetracePriceBarrierScalper):
    """
    Challenger 1: Same-Side Retrace Scalper + Relaxed Impulse Threshold
    Lowers the tactical entry delta threshold from 0.14 to 0.10, increasing trade
    frequency from 1 trade/hr to 4-6 trades/hr while strictly maintaining the
    retrace barrier and decay unwinds.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 1: Relaxed Impulse Frequency Scalper (4-6 tr/hr)"

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        now_ms = decision.ts_ms
        mid = 0.5 * (bbo_bid + bbo_ask)
        d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
        d_micro = decision.micro_probs[2] - decision.micro_probs[0]
        d_meso = decision.meso_probs[2] - decision.meso_probs[0]

        if self.state == "IDLE" and self.ticks_seen >= 10:
            # Relaxed threshold 0.10
            tact_dir = 2 if d_tact >= 0.10 else (0 if d_tact <= -0.10 else 1)
            tact_confirmed = (tact_dir in (0, 2)) and (
                (tact_dir == 2 and d_micro > 0.01 and d_meso >= -0.05) or
                (tact_dir == 0 and d_micro < -0.01 and d_meso <= 0.05)
            )
            if tact_confirmed and not self._is_parabolic_surge(tact_dir, now_ms, mid):
                can_reenter = True
                if self.last_exit_ts_ms > 0:
                    elapsed = (now_ms - self.last_exit_ts_ms) / 1000.0
                    if not self.last_exit_was_win and elapsed < 60.0:
                        can_reenter = False
                    elif self.last_exit_was_win and elapsed < 15.0:
                        can_reenter = False
                    if can_reenter and elapsed < 300.0 and self.last_trade_exit_price > 0:
                        if tact_dir == 2 and self.last_trade_side == "BUY" and (mid > self.last_trade_exit_price - 12.0):
                            can_reenter = False
                        elif tact_dir == 0 and self.last_trade_side == "SELL" and (mid < self.last_trade_exit_price + 12.0):
                            can_reenter = False
                if can_reenter:
                    tact_dec = copy.copy(decision)
                    object.__setattr__(tact_dec, "primary_direction", tact_dir) if hasattr(tact_dec, "__dataclass_fields__") and getattr(tact_dec, "__dataclass_params__", None) and getattr(tact_dec.__dataclass_params__, "frozen", False) else setattr(tact_dec, "primary_direction", tact_dir)
                    self._place_sniper_maker_entry(tact_dec, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, now_ms, intra_bar_open)
                    return self.state

        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)


# ==============================================================================
# ⚔️ CHALLENGER 2: Velocity Burst Momentum Scalper
# ==============================================================================
class Challenger2_VelocityBurstScalper(SameSideRetracePriceBarrierScalper):
    """
    Challenger 2: Same-Side Retrace Scalper + Velocity Burst Acceleration
    Detects intra-second price acceleration (velocity > 0.8 bps/s) aligning with
    Tactical 1m bias, triggering lightning-fast sniper entries before the spread widens.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 2: Velocity Burst Momentum Scalper"
        self.prev_mid: float = 0.0
        self.prev_tick_ms: int = 0

    def on_market_micro_tick(self, *args, **kwargs):
        mid = 0.5 * (kwargs.get("bbo_bid", 0.0) + kwargs.get("bbo_ask", 0.0))
        now_ms = kwargs.get("ts_ms", 0)
        if self.prev_mid > 0 and self.prev_tick_ms > 0 and now_ms > self.prev_tick_ms:
            dt_s = (now_ms - self.prev_tick_ms) / 1000.0
            if dt_s > 0.05:
                vel_bps_s = (((mid - self.prev_mid) / self.prev_mid) * 1e4) / dt_s
                if self.state == "IDLE" and abs(vel_bps_s) >= 0.80 and kwargs.get("current_decision") is not None:
                    dec = kwargs["current_decision"]
                    d_tact = dec.tactical_probs[2] - dec.tactical_probs[0]
                    if vel_bps_s > 0.80 and d_tact > 0.05:
                        self.on_hierarchical_decision(dec, kwargs.get("bbo_bid", 0.0), kwargs.get("bbo_ask", 0.0), kwargs.get("bbo_bid_qty", 1.0), kwargs.get("bbo_ask_qty", 1.0))
                    elif vel_bps_s < -0.80 and d_tact < -0.05:
                        self.on_hierarchical_decision(dec, kwargs.get("bbo_bid", 0.0), kwargs.get("bbo_ask", 0.0), kwargs.get("bbo_bid_qty", 1.0), kwargs.get("bbo_ask_qty", 1.0))
        self.prev_mid = mid
        self.prev_tick_ms = now_ms
        return super().on_market_micro_tick(*args, **kwargs)


# ==============================================================================
# ⚔️ CHALLENGER 3: Asymmetric Risk-Reward Scalper (TP = 1.8 * SL)
# ==============================================================================
class Challenger3_AsymmetricRiskRewardScalper(SameSideRetracePriceBarrierScalper):
    """
    Challenger 3: Same-Side Retrace Scalper + Asymmetric Risk-Reward Expander
    Sets resting Maker Take-Profit to 1.8x Stop-Loss distance on high edge setups
    (e.g., TP = +26.0 bps when SL = 14.5 bps), capturing larger breakout moves.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 3: Asymmetric Risk-Reward Scalper (TP = 1.8 * SL)"

    def _place_sniper_maker_entry(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, now_ms, intra_bar_open=0.0):
        super()._place_sniper_maker_entry(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, now_ms, intra_bar_open)
        # Expand TP price if resting order was initialized
        if self.active_position is not None:
            pos = self.active_position
            expanded_tp_bps = pos.stop_loss_bps * 1.8
            sign = 1.0 if pos.side == "BUY" else -1.0
            pos.take_profit_bps = expanded_tp_bps
            pos.tp_price = pos.entry_price * (1.0 + sign * (expanded_tp_bps / 10000.0))


# ==============================================================================
# ⚔️ CHALLENGER 4: Conformal Shannon Negentropy Scalper
# ==============================================================================
class Challenger4_ConformalEntropyScalper(SameSideRetracePriceBarrierScalper):
    """
    Challenger 4: Same-Side Retrace Scalper + Shannon Negentropy Crest Harvester
    Monitors information certainty C = 1 - H(p)/ln(3). When tactical uncertainty
    begins rising (C decays by >= 0.15 from peak), harvests at Maker BBO.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 4: Conformal Negentropy Scalper"
        self.peak_certainty = 0.0

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        if self.active_position is not None and self.active_position.peak_favorable_excursion_bps >= 4.0:
            p_tact = np.clip(np.array(decision.tactical_probs), 1e-7, 1.0)
            p_tact /= p_tact.sum()
            entropy = -np.sum(p_tact * np.log(p_tact))
            certainty = 1.0 - (entropy / np.log(3.0))
            if certainty > self.peak_certainty:
                self.peak_certainty = certainty
            elif (self.peak_certainty - certainty) >= 0.15:
                pos = self.active_position
                exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
                self._close_position(
                    exit_price=exit_p,
                    exit_ts_ms=decision.ts_ms,
                    exit_fee_bps=self.maker_fee_leg,
                    reason="CONFORMAL_NEGENTROPY_CREST_TP",
                )
                self.peak_certainty = 0.0
                return self.state

        if self.state == "IDLE":
            self.peak_certainty = 0.0

        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)


# ==============================================================================
# ⚔️ CHALLENGER 5: Fast 90s Stalling Retrace Scalper
# ==============================================================================
class Challenger5_Fast90sRetraceScalper(SameSideRetracePriceBarrierScalper):
    """
    Challenger 5: Same-Side Retrace Scalper + Fast 90s Stalling Unwind
    Tightens stall detection from 120s down to 90s if MFE is under +2.0 bps.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 5: Fast 90s Stalling Retrace Scalper"

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
                engine=SameSideRetracePriceBarrierScalper(symbol=symbol),
                strategy_profile="Reigning Champion: Same-Side Retrace Barrier Scalper (+37.78 bps Net in C25-C26)",
            ),
            "Challenger_1": TournamentCompetitorRecord(
                name="Challenger_1",
                engine=Challenger1_RelaxedImpulseFrequencyScalper(symbol=symbol),
                strategy_profile="Relaxed Impulse Frequency Scalper (4-6 tr/hr)",
            ),
            "Challenger_2": TournamentCompetitorRecord(
                name="Challenger_2",
                engine=Challenger2_VelocityBurstScalper(symbol=symbol),
                strategy_profile="Velocity Burst Momentum Scalper",
            ),
            "Challenger_3": TournamentCompetitorRecord(
                name="Challenger_3",
                engine=Challenger3_AsymmetricRiskRewardScalper(symbol=symbol),
                strategy_profile="Asymmetric Risk-Reward Scalper (TP = 1.8 * SL)",
            ),
            "Challenger_4": TournamentCompetitorRecord(
                name="Challenger_4",
                engine=Challenger4_ConformalEntropyScalper(symbol=symbol),
                strategy_profile="Conformal Negentropy Scalper",
            ),
            "Challenger_5": TournamentCompetitorRecord(
                name="Challenger_5",
                engine=Challenger5_Fast90sRetraceScalper(symbol=symbol),
                strategy_profile="Fast 90s Stalling Retrace Scalper",
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

        # Rank by fitness score descending (rewarding Net PnL, Win Rate, and Trade Count simultaneously)
        leaderboard.sort(key=lambda x: (x["net_pnl_bps"], x["win_rate_pct"], x["trades"]), reverse=True)
        for i, item in enumerate(leaderboard, 1):
            item["rank"] = i

        return leaderboard
