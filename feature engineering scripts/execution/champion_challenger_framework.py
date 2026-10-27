"""
feature engineering scripts/execution/champion_challenger_framework.py
Production Champion-Challenger Tournament Framework for Project-IMPJR Peak Trading System.

Cycle 28 Tournament Lineup:
  - 👑 Reigning Champion: Fast90sStallingRetraceScalper (Crowned Cycle 27 Winner: -8.44 bps net, 40% WR, 5 trades, superior 90s stall defense)
  - ⚔️ Challenger 1: WideBreatheTrailingScalper (Raises trailing hurdle to 8.0 bps, locks min 5.5 bps gross / +1.5 bps net, eliminates micro-choking)
  - ⚔️ Challenger 2: MultiTierTrendExpansionScalper (Expands TP to 2.2x SL on Macro/Meso concordance, holds for +30 bps wave)
  - ⚔️ Challenger 3: TacticalExhaustionPullbackScalper (Wide-breathe + exhaustion surge filter requiring 3 bps pullback before entry)
  - ⚔️ Challenger 4: MacroGuidedCrestScalper (Suppresses micro exits while Macro/Meso are trending strongly in trade direction)
  - ⚔️ Challenger 5: HighFrequencyWideHybridScalper (Wide-breathe + 0.12 tactical impulse threshold targeting 5-7 trades/hr with wide profits)
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
# BASELINE ENGINE: SameSideRetracePriceBarrierScalper
# ==============================================================================
class SameSideRetracePriceBarrierScalper(HierarchicalSniperExecutionEngine):
    """
    Combines High-Frequency Tactical Impulse Scaling with
    a Same-Side Retrace Price Barrier:
    When re-entering in the same direction within 300s, requires price to have retraced
    at least $15.00 better than the prior trade exit price.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Same-Side Retrace Barrier Baseline"
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
                    
                    # Same-Side Retrace Price Barrier check ($15 better than previous exit)
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
# 👑 REIGNING CHAMPION: Fast90sStallingRetraceScalper (Crowned Cycle 27 Winner)
# ==============================================================================
class Fast90sStallingRetraceScalper(SameSideRetracePriceBarrierScalper):
    """
    Reigning Champion (Crowned in Cycle 27):
    Combines the $15 Same-Side Retrace Price Barrier with 90s fast stall unwind.
    When a trade stalls for 90s without making progress (MFE < +2.0 bps), unwinds passively
    at Maker BBO to save critical drawdown.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Reigning Champion: Fast 90s Stalling Retrace Scalper (Cycle 27 Winner)"

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
# ⚔️ CHALLENGER 1: Wide-Breathe Trailing Scalper (Anti-Choking Solution)
# ==============================================================================
class Challenger1_WideBreatheTrailingScalper(Fast90sStallingRetraceScalper):
    """
    Challenger 1: Fast 90s Retrace Scalper + Wide-Breathe Trailing Shield
    Directly solves the micro-profit choking problem:
    - Does NOT arm trailing lock on minor 4.5 bps noise.
    - Arms trailing only once gross excursion reaches >= 8.0 bps (net >= +4.0 bps).
    - Locks a minimum of 5.50 bps gross (+1.50 bps net), expanding dynamically
      to lock +5.0 to +15.0 bps net as the trade expands into larger trends.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 1: Wide-Breathe Trailing Scalper (Anti-Choking)"
        self.min_trailing_activation_bps = 8.00
        self.tier1_min_lock_bps = 5.50
        self.model_harvest_min_bps = 7.50
        self.passive_unwind_min_bps = 5.50


# ==============================================================================
# ⚔️ CHALLENGER 2: Multi-Tier Trend Expansion Scalper (Macro/Meso Concordance)
# ==============================================================================
class Challenger2_MultiTierTrendExpansionScalper(Fast90sStallingRetraceScalper):
    """
    Challenger 2: Multi-Tier Trend Expansion Scalper
    Detects high-conviction concordance across Macro (15m) and Meso (5m).
    When Macro prob > 0.45 and Meso prob > 0.55, expands Take Profit from 25 bps
    to 32.0 bps (2.2x SL) and raises model harvest threshold to 9.0 bps gross,
    allowing the full multi-horizon expansion to reach crest profitability.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 2: Multi-Tier Trend Expansion Scalper"
        self.min_trailing_activation_bps = 9.00
        self.tier1_min_lock_bps = 6.00
        self.model_harvest_min_bps = 9.00

    def _place_sniper_maker_entry(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, now_ms, intra_bar_open=0.0):
        super()._place_sniper_maker_entry(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, now_ms, intra_bar_open)
        if self.active_position is not None:
            pos = self.active_position
            p_macro = decision.macro_probs[2] if pos.side == "BUY" else decision.macro_probs[0]
            p_meso = decision.meso_probs[2] if pos.side == "BUY" else decision.meso_probs[0]
            if p_macro > 0.45 and p_meso > 0.55:
                # Strong concordance: expand TP to 2.2x SL (31.9 bps)
                expanded_tp_bps = pos.stop_loss_bps * 2.2
                sign = 1.0 if pos.side == "BUY" else -1.0
                pos.take_profit_bps = expanded_tp_bps
                pos.tp_price = pos.entry_price * (1.0 + sign * (expanded_tp_bps / 10000.0))


# ==============================================================================
# ⚔️ CHALLENGER 3: Tactical Exhaustion Pullback Scalper
# ==============================================================================
class Challenger3_TacticalExhaustionPullbackScalper(Challenger1_WideBreatheTrailingScalper):
    """
    Challenger 3: Wide-Breathe + Tactical Exhaustion Surge Filter
    Prevents buying the exact exhaustion top (which caused Trades 4 & 5 to stall).
    If price moved > 12.0 bps in the proposed direction over the last 60s,
    requires at least a 3.0 bps micro-pullback before allowing Maker order placement.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 3: Tactical Exhaustion Pullback Scalper"
        self.price_history_60s: deque[tuple[int, float]] = deque(maxlen=7)

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        now_ms = decision.ts_ms
        mid = 0.5 * (bbo_bid + bbo_ask)
        self.price_history_60s.append((now_ms, mid))

        # Check if extended > 12 bps in 60s without 3 bps pullback
        if len(self.price_history_60s) >= 4:
            p_start = self.price_history_60s[0][1]
            p_max = max(p for _, p in self.price_history_60s)
            p_min = min(p for _, p in self.price_history_60s)
            surge_up_bps = ((p_max - p_start) / p_start) * 1e4
            surge_down_bps = ((p_start - p_min) / p_start) * 1e4
            pullback_from_high_bps = ((p_max - mid) / p_max) * 1e4
            pullback_from_low_bps = ((mid - p_min) / p_min) * 1e4

            d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
            if d_tact >= 0.14 and surge_up_bps > 12.0 and pullback_from_high_bps < 3.0:
                # Overextended upward: wait for at least 3 bps micro-dip
                return self.state
            elif d_tact <= -0.14 and surge_down_bps > 12.0 and pullback_from_low_bps < 3.0:
                # Overextended downward: wait for at least 3 bps micro-bounce
                return self.state

        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)


# ==============================================================================
# ⚔️ CHALLENGER 4: Macro-Guided Crest Scalper
# ==============================================================================
class Challenger4_MacroGuidedCrestScalper(Fast90sStallingRetraceScalper):
    """
    Challenger 4: Fast 90s Retrace Scalper + Macro-Guided Crest Harvester
    When in profit (>= 5.0 bps gross), suppresses micro exits if 15m Macro and 5m Meso
    are trending in full consensus with the trade (d_macro >= 0.10 and d_meso >= 0.10).
    Only harvests at full resting TP or when Meso begins decaying (d_meso < -0.05).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 4: Macro-Guided Crest Scalper"
        self.min_trailing_activation_bps = 7.50
        self.tier1_min_lock_bps = 5.20
        self.model_harvest_min_bps = 8.00

    def _manage_active_swing_position(self, decision, bbo_bid, bbo_ask, now_ms):
        if self.active_position is None:
            return
        pos = self.active_position
        sign = 1.0 if pos.side == "BUY" else -1.0
        d_macro = sign * (decision.macro_probs[2] - decision.macro_probs[0])
        d_meso = sign * (decision.meso_probs[2] - decision.meso_probs[0])

        # If Macro and Meso are solidly behind the trade, let it run and suppress premature exits
        if d_macro >= 0.10 and d_meso >= 0.10:
            # Let full resting TP or hard SL handle it
            mid = 0.5 * (bbo_bid + bbo_ask)
            unrealized_bps = sign * ((mid - pos.entry_price) / pos.entry_price) * 1e4
            if unrealized_bps >= pos.take_profit_bps:
                maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
                self._close_position(
                    exit_price=maker_exit_p,
                    exit_ts_ms=now_ms,
                    exit_fee_bps=self.maker_fee_leg,
                    reason="RESTING_MAKER_TP",
                )
                return
            return

        super()._manage_active_swing_position(decision, bbo_bid, bbo_ask, now_ms)


# ==============================================================================
# ⚔️ CHALLENGER 5: High-Frequency Wide Hybrid Scalper (5-7 trades/hr)
# ==============================================================================
class Challenger5_HighFrequencyWideHybridScalper(Challenger1_WideBreatheTrailingScalper):
    """
    Challenger 5: High-Frequency Wide Hybrid Scalper
    Combines Wide-Breathe Trailing (activation >= 8.0 bps, min lock 5.5 bps gross)
    with a calibrated tactical impulse entry threshold of 0.12 (lowered from 0.14)
    and 90s stall unwinding.
    Targets 5-7 trades/hour with high profit per trade.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 5: High-Frequency Wide Hybrid Scalper (5-7 tr/hr)"

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        now_ms = decision.ts_ms
        mid = 0.5 * (bbo_bid + bbo_ask)
        d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
        d_micro = decision.micro_probs[2] - decision.micro_probs[0]
        d_meso = decision.meso_probs[2] - decision.meso_probs[0]

        if self.state == "IDLE" and self.ticks_seen >= 10:
            # Calibrated 0.12 threshold for 5-7 trades/hr velocity
            tact_dir = 2 if d_tact >= 0.12 else (0 if d_tact <= -0.12 else 1)
            tact_confirmed = (tact_dir in (0, 2)) and (
                (tact_dir == 2 and d_micro > 0.015 and d_meso >= -0.05) or
                (tact_dir == 0 and d_micro < -0.015 and d_meso <= 0.05)
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
                engine=Fast90sStallingRetraceScalper(symbol=symbol),
                strategy_profile="Reigning Champion: Fast 90s Stalling Retrace Scalper (Cycle 27 Winner)",
            ),
            "Challenger_1": TournamentCompetitorRecord(
                name="Challenger_1",
                engine=Challenger1_WideBreatheTrailingScalper(symbol=symbol),
                strategy_profile="Wide-Breathe Trailing Scalper (Anti-Choking)",
            ),
            "Challenger_2": TournamentCompetitorRecord(
                name="Challenger_2",
                engine=Challenger2_MultiTierTrendExpansionScalper(symbol=symbol),
                strategy_profile="Multi-Tier Trend Expansion Scalper",
            ),
            "Challenger_3": TournamentCompetitorRecord(
                name="Challenger_3",
                engine=Challenger3_TacticalExhaustionPullbackScalper(symbol=symbol),
                strategy_profile="Tactical Exhaustion Pullback Scalper",
            ),
            "Challenger_4": TournamentCompetitorRecord(
                name="Challenger_4",
                engine=Challenger4_MacroGuidedCrestScalper(symbol=symbol),
                strategy_profile="Macro-Guided Crest Scalper",
            ),
            "Challenger_5": TournamentCompetitorRecord(
                name="Challenger_5",
                engine=Challenger5_HighFrequencyWideHybridScalper(symbol=symbol),
                strategy_profile="High-Frequency Wide Hybrid Scalper (5-7 tr/hr)",
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
