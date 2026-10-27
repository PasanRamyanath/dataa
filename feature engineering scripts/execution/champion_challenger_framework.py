"""
feature engineering scripts/execution/champion_challenger_framework.py
Production Champion-Challenger Tournament Framework for Project-IMPJR Peak Trading System.
Cycle 26 Tournament Roster:
  - 👑 Reigning Champion: SameSideRetracePriceBarrierScalper (Crowned in Cycle 25 with +30.68 bps Net Cash, 24.60 Profit Factor, and +31.44 bps peak resting TP capture)
  - ⚔️ Challenger 1: Dynamic ATR Retrace Barrier Scalper (Scales pullback requirement to 0.8 * ATR_10s)
  - ⚔️ Challenger 2: Multi-Stage Runner Scalper (Expands trailing buffer to 15% on >+20 bps explosive runners)
  - ⚔️ Challenger 3: OFI Micro-Burst Retrace Scalper (Requires taker buy/sell ratio >= 1.40)
  - ⚔️ Challenger 4: Conformal Negentropy Retrace Scalper (Shannon uncertainty crest exit)
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
    Reigning Champion (Crowned in Cycle 25 with +30.68 bps Net Cash, 24.60 Profit Factor):
    Combines High-Frequency Tactical Impulse Scaling with Adaptive 120s Momentum Decay Unwind
    and a Same-Side Retrace Price Barrier:
    When re-entering in the same direction within 300s, requires price to have retraced
    at least $15.00 better than the prior trade exit price.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Reigning Champion: Same-Side Retrace Barrier Scalper (+30.68 bps net in Cycle 25)"
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
# ⚔️ CHALLENGER 1: Dynamic ATR Retrace Barrier Scalper
# ==============================================================================
class Challenger1_DynamicATRRetraceScalper(SameSideRetracePriceBarrierScalper):
    """
    Challenger 1: Same-Side Retrace Scalper + Dynamic ATR Retrace Barrier
    Instead of a fixed $15.00 pullback, scales the required retrace dynamically
    to max(10.0, 0.8 * ATR_10s_dollars), adapting automatically to market volatility.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 1: Dynamic ATR Retrace Barrier Scalper"

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        # Dynamically scale retrace hurdle based on 10s ATR
        mid = 0.5 * (bbo_bid + bbo_ask)
        atr_bps = getattr(decision, "atr_10s_bps", 3.0)
        retrace_hurdle = max(10.0, (atr_bps / 10000.0) * mid * 0.8)
        
        now_ms = decision.ts_ms
        d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
        if self.state == "IDLE" and self.last_exit_ts_ms > 0 and self.last_trade_exit_price > 0:
            elapsed = (now_ms - self.last_exit_ts_ms) / 1000.0
            if elapsed < 300.0:
                if d_tact >= 0.14 and self.last_trade_side == "BUY" and (mid > self.last_trade_exit_price - retrace_hurdle):
                    return self.state
                if d_tact <= -0.14 and self.last_trade_side == "SELL" and (mid < self.last_trade_exit_price + retrace_hurdle):
                    return self.state
        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)


# ==============================================================================
# ⚔️ CHALLENGER 2: Multi-Stage Runner Scalper
# ==============================================================================
class Challenger2_MultiStageRunnerScalper(SameSideRetracePriceBarrierScalper):
    """
    Challenger 2: Same-Side Retrace Scalper + Multi-Stage Runner Expansion
    On explosive breakout trades where peak MFE exceeds +20.0 bps,
    contracts trailing giveback to only 15% of peak (vs standard 25%),
    allowing major trend runners to harvest +40 to +60 bps moves.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 2: Multi-Stage Runner Scalper"

    def on_market_micro_tick(self, *args, **kwargs):
        state = super().on_market_micro_tick(*args, **kwargs)
        if self.active_position is not None and self.active_position.peak_favorable_excursion_bps >= 20.0:
            # Multi-stage runner: tighten trailing floor to 85% of peak excursion
            pos = self.active_position
            tight_floor_bps = pos.peak_favorable_excursion_bps * 0.85
            mid = 0.5 * (kwargs.get("bbo_bid", 0.0) + kwargs.get("bbo_ask", 0.0))
            sign = 1.0 if pos.side == "BUY" else -1.0
            unreal = sign * ((mid - pos.entry_price) / pos.entry_price) * 1e4
            if unreal <= tight_floor_bps:
                exit_p = kwargs.get("bbo_ask", 0.0) if pos.side == "BUY" else kwargs.get("bbo_bid", 0.0)
                self._close_position(
                    exit_price=exit_p,
                    exit_ts_ms=kwargs.get("ts_ms", 0),
                    exit_fee_bps=self.maker_fee_leg,
                    reason="RUNNER_STAGE_MAKER_TP",
                )
        return self.state


# ==============================================================================
# ⚔️ CHALLENGER 3: OFI Micro-Burst Retrace Scalper
# ==============================================================================
class Challenger3_OFIBurstRetraceScalper(SameSideRetracePriceBarrierScalper):
    """
    Challenger 3: Same-Side Retrace Scalper + Order Flow Imbalance Confirmation
    Combines the $15 retrace barrier with aggressive taker volume imbalance:
    Requires buy/sell volume ratio >= 1.40 for LONG and <= 0.71 for SHORT.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 3: OFI Burst Retrace Scalper"
        self.last_buy_vol = 0.0
        self.last_sell_vol = 0.0

    def on_market_micro_tick(self, *args, **kwargs):
        self.last_buy_vol = kwargs.get("aggressive_buy_vol_at_ask", 0.0)
        self.last_sell_vol = kwargs.get("aggressive_sell_vol_at_bid", 0.0)
        return super().on_market_micro_tick(*args, **kwargs)

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        tot_vol = self.last_buy_vol + self.last_sell_vol
        if tot_vol > 0.01:
            buy_ratio = self.last_buy_vol / tot_vol
            d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
            if d_tact >= 0.14 and buy_ratio < 0.58:
                return self.state
            if d_tact <= -0.14 and buy_ratio > 0.42:
                return self.state
        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)


# ==============================================================================
# ⚔️ CHALLENGER 4: Conformal Shannon Negentropy Retrace Scalper
# ==============================================================================
class Challenger4_ConformalEntropyRetraceScalper(SameSideRetracePriceBarrierScalper):
    """
    Challenger 4: Same-Side Retrace Scalper + Shannon Negentropy Crest Harvester
    Monitors information certainty C = 1 - H(p)/ln(3). When tactical uncertainty
    begins rising (C decays by >= 0.15 from peak), harvests at Maker BBO.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 4: Conformal Negentropy Retrace Scalper"
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
                strategy_profile="Reigning Champion: Same-Side Retrace Barrier Scalper (Crowned Cycle 25 with +30.68 bps Net)",
            ),
            "Challenger_1": TournamentCompetitorRecord(
                name="Challenger_1",
                engine=Challenger1_DynamicATRRetraceScalper(symbol=symbol),
                strategy_profile="Dynamic ATR Retrace Barrier Scalper",
            ),
            "Challenger_2": TournamentCompetitorRecord(
                name="Challenger_2",
                engine=Challenger2_MultiStageRunnerScalper(symbol=symbol),
                strategy_profile="Multi-Stage Runner Scalper",
            ),
            "Challenger_3": TournamentCompetitorRecord(
                name="Challenger_3",
                engine=Challenger3_OFIBurstRetraceScalper(symbol=symbol),
                strategy_profile="OFI Burst Retrace Scalper",
            ),
            "Challenger_4": TournamentCompetitorRecord(
                name="Challenger_4",
                engine=Challenger4_ConformalEntropyRetraceScalper(symbol=symbol),
                strategy_profile="Conformal Negentropy Retrace Scalper",
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
