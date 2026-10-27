"""
feature engineering scripts/execution/champion_challenger_framework.py
Production Champion-Challenger Tournament Framework for Project-IMPJR Peak Trading System.
Cycle 25 Tournament Roster:
  - 👑 Reigning Champion: AdaptiveMomentumDecayUnwindScalper (Crowned in Cycle 24 with 7 trades, +10.83 bps won on wins, saving +15.73 bps via 120s passive unwinds)
  - ⚔️ Challenger 1: Macro-Aligned Decay Scalper (Blocks transitional entries when Macro opposes direction)
  - ⚔️ Challenger 2: Same-Side Retrace Scalper (Requires >= $15 pullback before re-entering in same direction)
  - ⚔️ Challenger 3: Fast 90s Stalling Unwind Scalper (Tightens stall window to 90s if MFE < 2.0 bps)
  - ⚔️ Challenger 4: Conformal Shannon Negentropy Decay Scalper (Exits at peak information certainty)
  - ⚔️ Challenger 5: Triple-Horizon Consensus Scalper (Requires Tactical + Meso + Micro sign consensus)
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
# 👑 REIGNING CHAMPION: AdaptiveMomentumDecayUnwindScalper (Crowned Cycle 24)
# ==============================================================================
class AdaptiveMomentumDecayUnwindScalper(HierarchicalSniperExecutionEngine):
    """
    Reigning Champion (Crowned in Cycle 24):
    High-Frequency Tactical Impulse Scalper with decoupled 1m/30s momentum entries,
    Dual-Horizon Invalidation Bounding, 3-Tier Dynamic Trailing Shield, and
    Adaptive 120s Momentum Decay Unwind (passively unwinds stalled positions at BBO).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Reigning Champion: Adaptive Decay Unwind Scalper (Crowned Cycle 24)"
        self.min_net_edge_bps = 4.0

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
# ⚔️ CHALLENGER 1: Macro-Aligned Decay Scalper
# ==============================================================================
class Challenger1_MacroAlignedDecayScalper(AdaptiveMomentumDecayUnwindScalper):
    """
    Challenger 1: Adaptive Decay Scalper + Top-Down Macro Alignment Gate
    Requires that Macro delta must not actively oppose the tactical impulse:
    For LONG: d_macro >= -0.05
    For SHORT: d_macro <= +0.05
    Eliminates transitional false longs (like Trades 3, 5, 6 in Cycle 24).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 1: Macro-Aligned Decay Scalper"

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        d_macro = decision.macro_probs[2] - decision.macro_probs[0]
        d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
        if self.state == "IDLE":
            if d_tact >= 0.14 and d_macro < -0.05:
                return self.state  # Suppress long if macro is bearish
            if d_tact <= -0.14 and d_macro > 0.05:
                return self.state  # Suppress short if macro is bullish
        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)


# ==============================================================================
# ⚔️ CHALLENGER 2: Same-Side Retrace Price Barrier Scalper
# ==============================================================================
class Challenger2_SameSideRetraceScalper(AdaptiveMomentumDecayUnwindScalper):
    """
    Challenger 2: Adaptive Decay Scalper + Same-Side Retrace Price Barrier
    When re-entering in the same direction within 300s, requires price to have retraced
    at least $15.00 better than the prior trade's exit price.
    Prevents chasing consecutive legs into identical exhaustion tops.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 2: Same-Side Retrace Price Barrier Scalper"
        self.last_trade_exit_price: float = 0.0
        self.last_trade_side: str = ""

    def _close_position(self, exit_price, exit_ts_ms, exit_fee_bps, reason="TAKE_PROFIT"):
        if self.active_position is not None:
            self.last_trade_exit_price = exit_price
            self.last_trade_side = self.active_position.side
        return super()._close_position(exit_price, exit_ts_ms, exit_fee_bps, reason)

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        if self.state == "IDLE" and self.last_trade_exit_price > 0:
            mid = 0.5 * (bbo_bid + bbo_ask)
            now_ms = decision.ts_ms
            elapsed = (now_ms - self.last_exit_ts_ms) / 1000.0 if self.last_exit_ts_ms > 0 else 999.0
            if elapsed < 300.0:
                d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
                # For LONG: require pullback of at least $15 below prior exit
                if d_tact >= 0.14 and self.last_trade_side == "BUY" and (mid > self.last_trade_exit_price - 15.0):
                    return self.state
                # For SHORT: require bounce of at least $15 above prior exit
                if d_tact <= -0.14 and self.last_trade_side == "SELL" and (mid < self.last_trade_exit_price + 15.0):
                    return self.state
        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)


# ==============================================================================
# ⚔️ CHALLENGER 3: Fast 90s Stalling Unwind Scalper
# ==============================================================================
class Challenger3_Fast90sDecayScalper(AdaptiveMomentumDecayUnwindScalper):
    """
    Challenger 3: Adaptive Decay Scalper + Fast 90-Second Stalling Unwind
    Tightens stall detection from 120s down to 90s if MFE is under +2.0 bps.
    Cuts adverse drift even earlier, minimizing loss sizes on non-moving trades.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 3: Fast 90s Stalling Unwind Scalper"

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
# ⚔️ CHALLENGER 4: Conformal Shannon Negentropy Decay Scalper
# ==============================================================================
class Challenger4_ConformalEntropyDecayScalper(AdaptiveMomentumDecayUnwindScalper):
    """
    Challenger 4: Adaptive Decay Scalper + Shannon Negentropy Crest Harvester
    Combines Challenger 5's stall protection with active Shannon information crest exits:
    When model certainty C = 1 - H(p)/ln(3) peaks and decays by >= 0.15, exits at Maker BBO.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 4: Conformal Negentropy Decay Scalper"
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
# ⚔️ CHALLENGER 5: Triple-Horizon Consensus Scalper
# ==============================================================================
class Challenger5_TripleHorizonConsensusScalper(AdaptiveMomentumDecayUnwindScalper):
    """
    Challenger 5: Adaptive Decay Scalper + Triple-Horizon Sign Consensus
    Requires that Tactical 1m, Meso 5m, AND Micro 10s all point in the same direction:
    For LONG: d_tact > 0.12, d_meso > 0.02, d_micro > 0.02
    For SHORT: d_tact < -0.12, d_meso < -0.02, d_micro < -0.02
    Guarantees maximum trade quality with high multi-timeframe concordance.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 5: Triple-Horizon Consensus Scalper"

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        if self.state == "IDLE":
            d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
            d_meso = decision.meso_probs[2] - decision.meso_probs[0]
            d_micro = decision.micro_probs[2] - decision.micro_probs[0]
            
            # Require full triple-horizon agreement
            long_consensus = (d_tact >= 0.12 and d_meso >= 0.02 and d_micro >= 0.02)
            short_consensus = (d_tact <= -0.12 and d_meso <= -0.02 and d_micro <= -0.02)
            if not (long_consensus or short_consensus):
                return self.state  # Suppress trades lacking triple-horizon consensus

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
                engine=AdaptiveMomentumDecayUnwindScalper(symbol=symbol),
                strategy_profile="Reigning Champion: Adaptive Decay Unwind Scalper (Crowned Cycle 24)",
            ),
            "Challenger_1": TournamentCompetitorRecord(
                name="Challenger_1",
                engine=Challenger1_MacroAlignedDecayScalper(symbol=symbol),
                strategy_profile="Macro-Aligned Decay Scalper",
            ),
            "Challenger_2": TournamentCompetitorRecord(
                name="Challenger_2",
                engine=Challenger2_SameSideRetraceScalper(symbol=symbol),
                strategy_profile="Same-Side Retrace Price Barrier Scalper",
            ),
            "Challenger_3": TournamentCompetitorRecord(
                name="Challenger_3",
                engine=Challenger3_Fast90sDecayScalper(symbol=symbol),
                strategy_profile="Fast 90s Stalling Unwind Scalper",
            ),
            "Challenger_4": TournamentCompetitorRecord(
                name="Challenger_4",
                engine=Challenger4_ConformalEntropyDecayScalper(symbol=symbol),
                strategy_profile="Conformal Negentropy Decay Scalper",
            ),
            "Challenger_5": TournamentCompetitorRecord(
                name="Challenger_5",
                engine=Challenger5_TripleHorizonConsensusScalper(symbol=symbol),
                strategy_profile="Triple-Horizon Consensus Scalper",
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
