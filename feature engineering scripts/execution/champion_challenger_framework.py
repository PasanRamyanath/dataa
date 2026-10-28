"""
feature engineering scripts/execution/champion_challenger_framework.py
Production Champion-Challenger Tournament Framework for Project-IMPJR Peak Trading System.

Cycle 31 Tournament Lineup (Peak System Restored):
  - 👑 Reigning Champion: PeakSameSideRetraceScalper (Restored Undefeated C25-C26 Baseline with +37.78 bps Net, 100% WR in C26, and Positive Break-Even Shield)
  - ⚔️ Challenger 1: StrictMesoConcordanceScalper (Enforces d_meso >= +0.02 for BUY and <= -0.02 for SELL to eliminate counter-trend knife catches)
  - ⚔️ Challenger 2: ExtremeRangeBoundaryFilterScalper (15m rolling envelope filter: blocks SELL near session floor and BUY near session ceiling)
  - ⚔️ Challenger 3: AsymmetricRiskRewardScalper (Expands resting Maker TP to 35 bps gross with 3:1 RR on strong macro/meso consensus)
  - ⚔️ Challenger 4: WideBreatheHarvestScalper (Wider 8.5 bps trailing arming and 7.5 bps model harvest to ride +30 bps runners like C25 Trade 3 & C29 Trade 5)
  - ⚔️ Challenger 5: DynamicFrequencyPeakScalper (Calibrated 0.12 impulse threshold + $14 retrace barrier for 4-6 high-conviction trades/hr)

Tournament Gating Rule:
  An absolute profitability gate is enforced. A Challenger can ONLY dethrone the Champion if:
  1. Net Realized PnL is strictly positive (> 0.0 bps)
  2. Win Rate is at least 75.0%
  3. Net PnL is greater than Reigning Champion's net PnL
  Losing strategies (net PnL <= 0 or WR < 75%) are permanently rejected and never promoted to Champion!
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
# 👑 REIGNING CHAMPION: Peak Same-Side Retrace Scalper (Crowned Cycle 25 & 26)
# ==============================================================================
class PeakSameSideRetraceScalper(HierarchicalSniperExecutionEngine):
    """
    Undefeated Champion (Crowned in Cycle 25 with +30.68 bps Net, defended in Cycle 26 with +7.10 bps Net, 100% WR):
    Combines:
    1. Multi-Horizon Stage 5 Supervised Fusion + High-Frequency Tactical Impulse Scaling.
    2. Meso Concordance Guard (d_meso >= 0.00 for BUY, d_meso <= 0.00 for SELL).
    3. Same-Side Retrace Price Barrier: Requires $15.00 pullback before re-entering same side within 300s.
    4. Positive Break-Even Profit Shield: Locks +4.10 bps gross (+0.10 bps net profit guaranteed) once MFE >= 3.80 bps.
    5. Wide-Breathe Trailing & Neural Model Profit Harvest (MODEL_HARVEST_MAKER_TP).
    6. NO aggressive 90s stall cut (allows trades full 5m swing development).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Reigning Champion: Peak Same-Side Retrace Scalper (+37.78 bps Net in C25-C26)"
        self.min_trailing_activation_bps = 7.00
        self.tier1_min_lock_bps = 4.50
        self.model_harvest_min_bps = 5.00
        self.passive_unwind_min_bps = 4.25
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
            cand_dir = None
            tact_dir = 2 if d_tact >= 0.14 else (0 if d_tact <= -0.14 else 1)
            tact_confirmed = (tact_dir in (0, 2)) and (
                (tact_dir == 2 and d_micro > 0.02 and d_meso >= 0.00) or
                (tact_dir == 0 and d_micro < -0.02 and d_meso <= 0.00)
            )

            if tact_confirmed:
                cand_dir = tact_dir
            elif decision.primary_direction in (0, 2) and decision.expected_net_edge_bps >= self.min_net_edge_bps:
                if decision.primary_direction == 2 and d_meso >= 0.00:
                    cand_dir = 2
                elif decision.primary_direction == 0 and d_meso <= 0.00:
                    cand_dir = 0

            if cand_dir in (0, 2):
                # 1. Parabolic Surge Guard (Micro-spike / Tactical wave / Meso wave exhaustion veto)
                if self._is_parabolic_surge(cand_dir, now_ms, mid):
                    return self.state

                # 2. Strict Retrace Barrier & Cooldown
                if self.last_exit_ts_ms > 0:
                    elapsed = (now_ms - self.last_exit_ts_ms) / 1000.0
                    if not self.last_exit_was_win and elapsed < 90.0:
                        return self.state
                    elif self.last_exit_was_win and elapsed < 20.0:
                        return self.state

                    # Same-Side Retrace Price Barrier: Requires $15.00 pullback from previous exit
                    if elapsed < 300.0 and self.last_trade_exit_price > 0:
                        if cand_dir == 2 and self.last_trade_side == "BUY" and (mid > self.last_trade_exit_price - 15.0):
                            return self.state
                        elif cand_dir == 0 and self.last_trade_side == "SELL" and (mid < self.last_trade_exit_price + 15.0):
                            return self.state

                # Authorized Peak Sniper Maker Entry!
                tact_dec = copy.copy(decision)
                object.__setattr__(tact_dec, "primary_direction", cand_dir) if hasattr(tact_dec, "__dataclass_fields__") and getattr(tact_dec, "__dataclass_params__", None) and getattr(tact_dec.__dataclass_params__, "frozen", False) else setattr(tact_dec, "primary_direction", cand_dir)
                self.stalking_decision = tact_dec
                self._place_sniper_maker_entry(tact_dec, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, now_ms, intra_bar_open)
                return self.state

        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)


# ==============================================================================
# ⚔️ CHALLENGER 1: Strict Meso Concordance Scalper (Anti-Hostile Flow)
# ==============================================================================
class Challenger1_StrictMesoConcordanceScalper(PeakSameSideRetraceScalper):
    """
    Challenger 1: Strict Meso Concordance Scalper
    Tightens Meso concordance filter:
    Requires d_meso >= +0.02 for BUY and d_meso <= -0.02 for SELL.
    Completely eliminates hostile counter-trend trades during adverse 5-minute flows.
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
            cand_dir = None
            tact_dir = 2 if d_tact >= 0.14 else (0 if d_tact <= -0.14 else 1)
            tact_confirmed = (tact_dir in (0, 2)) and (
                (tact_dir == 2 and d_micro > 0.02 and d_meso >= 0.02) or
                (tact_dir == 0 and d_micro < -0.02 and d_meso <= -0.02)
            )

            if tact_confirmed:
                cand_dir = tact_dir
            elif decision.primary_direction in (0, 2) and decision.expected_net_edge_bps >= self.min_net_edge_bps:
                if decision.primary_direction == 2 and d_meso >= 0.02:
                    cand_dir = 2
                elif decision.primary_direction == 0 and d_meso <= -0.02:
                    cand_dir = 0

            if cand_dir in (0, 2):
                if self._is_parabolic_surge(cand_dir, now_ms, mid):
                    return self.state

                if self.last_exit_ts_ms > 0:
                    elapsed = (now_ms - self.last_exit_ts_ms) / 1000.0
                    if not self.last_exit_was_win and elapsed < 90.0:
                        return self.state
                    elif self.last_exit_was_win and elapsed < 20.0:
                        return self.state

                    if elapsed < 300.0 and self.last_trade_exit_price > 0:
                        if cand_dir == 2 and self.last_trade_side == "BUY" and (mid > self.last_trade_exit_price - 15.0):
                            return self.state
                        elif cand_dir == 0 and self.last_trade_side == "SELL" and (mid < self.last_trade_exit_price + 15.0):
                            return self.state

                tact_dec = copy.copy(decision)
                object.__setattr__(tact_dec, "primary_direction", cand_dir) if hasattr(tact_dec, "__dataclass_fields__") and getattr(tact_dec, "__dataclass_params__", None) and getattr(tact_dec.__dataclass_params__, "frozen", False) else setattr(tact_dec, "primary_direction", cand_dir)
                self.stalking_decision = tact_dec
                self._place_sniper_maker_entry(tact_dec, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, now_ms, intra_bar_open)
                return self.state

        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)


# ==============================================================================
# ⚔️ CHALLENGER 2: Extreme Range Boundary Filter Scalper
# ==============================================================================
class Challenger2_ExtremeRangeBoundaryFilterScalper(PeakSameSideRetraceScalper):
    """
    Challenger 2: Extreme Range Boundary Filter Scalper
    Maintains a rolling 15-minute price envelope (90 bars).
    Blocks SELL orders within $15.00 of the 15m session low (anti-bottom-selling).
    Blocks BUY orders within $15.00 of the 15m session high (anti-top-buying).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 2: Extreme Range Boundary Filter Scalper"
        self.rolling_15m_prices: deque[tuple[int, float]] = deque(maxlen=90)

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        now_ms = decision.ts_ms
        mid = 0.5 * (bbo_bid + bbo_ask)
        self.rolling_15m_prices.append((now_ms, mid))

        if len(self.rolling_15m_prices) >= 18 and self.state == "IDLE":
            session_high = max(p for _, p in self.rolling_15m_prices)
            session_low = min(p for _, p in self.rolling_15m_prices)
            d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]

            # Range boundary filters
            if d_tact <= -0.14 and (mid <= session_low + 15.0):
                return self.state
            elif d_tact >= 0.14 and (mid >= session_high - 15.0):
                return self.state

        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)


# ==============================================================================
# ⚔️ CHALLENGER 3: Asymmetric Risk-Reward Scalper (3:1 RR Target)
# ==============================================================================
class Challenger3_AsymmetricRiskRewardScalper(PeakSameSideRetraceScalper):
    """
    Challenger 3: Asymmetric Risk-Reward Scalper (3:1 RR Target)
    When higher timeframes agree (Macro > 0.38 and Meso > 0.42), expands resting Maker TP
    to 35.0 bps gross (+31.0 bps net) and tightens stop loss to 12.0 bps.
    Combines with positive break-even profit shield to lock banked profits.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 3: Asymmetric Risk-Reward Scalper (3:1 RR Target)"

    def _place_sniper_maker_entry(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, now_ms, intra_bar_open=0.0):
        super()._place_sniper_maker_entry(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, now_ms, intra_bar_open)
        if self.active_position is not None:
            pos = self.active_position
            p_macro = decision.macro_probs[2] if pos.side == "BUY" else decision.macro_probs[0]
            p_meso = decision.meso_probs[2] if pos.side == "BUY" else decision.meso_probs[0]
            if p_macro > 0.38 and p_meso > 0.42:
                expanded_tp_bps = 35.0
                tightened_sl_bps = 12.0
                sign = 1.0 if pos.side == "BUY" else -1.0
                pos.take_profit_bps = expanded_tp_bps
                pos.stop_loss_bps = tightened_sl_bps
                pos.tp_price = pos.entry_price * (1.0 + sign * (expanded_tp_bps / 10000.0))
                pos.sl_price = pos.entry_price * (1.0 - sign * (tightened_sl_bps / 10000.0))


# ==============================================================================
# ⚔️ CHALLENGER 4: Wide-Breathe Harvest Scalper (+30 bps Runner Target)
# ==============================================================================
class Challenger4_WideBreatheHarvestScalper(PeakSameSideRetraceScalper):
    """
    Challenger 4: Wide-Breathe Harvest Scalper
    Widens trailing activation hurdle to 8.50 bps and model harvest threshold to 7.50 bps.
    Designed specifically to let high-conviction breakout swings run to full +25 to +35 bps
    resting TP capture (preserving the Trade 5 edge of Cycle 29).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 4: Wide-Breathe Harvest Scalper (+30 bps Runner Target)"
        self.min_trailing_activation_bps = 8.50
        self.tier1_min_lock_bps = 5.50
        self.model_harvest_min_bps = 7.50
        self.passive_unwind_min_bps = 5.50


# ==============================================================================
# ⚔️ CHALLENGER 5: Dynamic Frequency Peak Scalper (4-6 tr/hr Target)
# ==============================================================================
class Challenger5_DynamicFrequencyPeakScalper(Challenger2_ExtremeRangeBoundaryFilterScalper):
    """
    Challenger 5: Dynamic Frequency Peak Scalper
    Calibrated 0.12 tactical impulse threshold for 4-6 trades/hr frequency
    while enforcing strict Meso concordance (d_meso >= 0.00 for BUY, <= 0.00 for SELL)
    and the extreme range-boundary support/resistance envelope filter.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 5: Dynamic Frequency Peak Scalper (4-6 tr/hr Target)"

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        now_ms = decision.ts_ms
        mid = 0.5 * (bbo_bid + bbo_ask)
        d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
        d_micro = decision.micro_probs[2] - decision.micro_probs[0]
        d_meso = decision.meso_probs[2] - decision.meso_probs[0]

        if self.state == "IDLE" and self.ticks_seen >= 10:
            tact_dir = 2 if d_tact >= 0.12 else (0 if d_tact <= -0.12 else 1)
            tact_confirmed = (tact_dir in (0, 2)) and (
                (tact_dir == 2 and d_micro > 0.015 and d_meso >= 0.00) or
                (tact_dir == 0 and d_micro < -0.015 and d_meso <= 0.00)
            )

            cand_dir = tact_dir if tact_confirmed else None
            if cand_dir is None and decision.primary_direction in (0, 2) and decision.expected_net_edge_bps >= self.min_net_edge_bps:
                if decision.primary_direction == 2 and d_meso >= 0.00:
                    cand_dir = 2
                elif decision.primary_direction == 0 and d_meso <= 0.00:
                    cand_dir = 0

            if cand_dir in (0, 2):
                if self._is_parabolic_surge(cand_dir, now_ms, mid):
                    return self.state

                if len(self.rolling_15m_prices) >= 18:
                    session_high = max(p for _, p in self.rolling_15m_prices)
                    session_low = min(p for _, p in self.rolling_15m_prices)
                    if cand_dir == 0 and (mid <= session_low + 14.0):
                        return self.state
                    elif cand_dir == 2 and (mid >= session_high - 14.0):
                        return self.state

                if self.last_exit_ts_ms > 0:
                    elapsed = (now_ms - self.last_exit_ts_ms) / 1000.0
                    if not self.last_exit_was_win and elapsed < 75.0:
                        return self.state
                    elif self.last_exit_was_win and elapsed < 15.0:
                        return self.state

                    if elapsed < 300.0 and self.last_trade_exit_price > 0:
                        if cand_dir == 2 and self.last_trade_side == "BUY" and (mid > self.last_trade_exit_price - 14.0):
                            return self.state
                        elif cand_dir == 0 and self.last_trade_side == "SELL" and (mid < self.last_trade_exit_price + 14.0):
                            return self.state

                tact_dec = copy.copy(decision)
                object.__setattr__(tact_dec, "primary_direction", cand_dir) if hasattr(tact_dec, "__dataclass_fields__") and getattr(tact_dec, "__dataclass_params__", None) and getattr(tact_dec.__dataclass_params__, "frozen", False) else setattr(tact_dec, "primary_direction", cand_dir)
                self.stalking_decision = tact_dec
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
    Maintains the live leaderboard and enforces the ABSOLUTE PROFITABILITY GATE:
    A challenger can only dethrone the champion if net_pnl_bps > 0.0 and win_rate_pct >= 75.0%.
    """
    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol
        self.competitors: dict[str, TournamentCompetitorRecord] = {
            "Champion": TournamentCompetitorRecord(
                name="Champion",
                engine=PeakSameSideRetraceScalper(symbol=symbol),
                strategy_profile="Reigning Champion: Peak Same-Side Retrace Scalper (+37.78 bps Net in C25-C26)",
            ),
            "Challenger_1": TournamentCompetitorRecord(
                name="Challenger_1",
                engine=Challenger1_StrictMesoConcordanceScalper(symbol=symbol),
                strategy_profile="Strict Meso Concordance Scalper (Anti-Hostile Flow)",
            ),
            "Challenger_2": TournamentCompetitorRecord(
                name="Challenger_2",
                engine=Challenger2_ExtremeRangeBoundaryFilterScalper(symbol=symbol),
                strategy_profile="Extreme Range Boundary Filter Scalper",
            ),
            "Challenger_3": TournamentCompetitorRecord(
                name="Challenger_3",
                engine=Challenger3_AsymmetricRiskRewardScalper(symbol=symbol),
                strategy_profile="Asymmetric Risk-Reward Scalper (3:1 RR Target)",
            ),
            "Challenger_4": TournamentCompetitorRecord(
                name="Challenger_4",
                engine=Challenger4_WideBreatheHarvestScalper(symbol=symbol),
                strategy_profile="Wide-Breathe Harvest Scalper (+30 bps Runner Target)",
            ),
            "Challenger_5": TournamentCompetitorRecord(
                name="Challenger_5",
                engine=Challenger5_DynamicFrequencyPeakScalper(symbol=symbol),
                strategy_profile="Dynamic Frequency Peak Scalper (4-6 tr/hr Target)",
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

    def determine_tournament_winner(self, reigning_champ_name: str = "Champion") -> dict[str, Any]:
        """
        Determines the legitimate tournament winner using the ABSOLUTE PEAK PERFORMANCE GATE:
        1. A Challenger can ONLY dethrone the Champion if:
           - Challenger's net PnL is STRICTLY POSITIVE (> 0.0 bps)
           - Challenger's win rate is at least 75.0%
           - Challenger's net PnL is strictly greater than the Reigning Champion's net PnL
        2. If the Reigning Champion is profitable (> 0.0 bps) and has WR >= 75%, and no Challenger
           beats it with strictly higher positive PnL and WR >= 75%, the Reigning Champion DEFENDS its crown!
        3. If NO competitor achieved positive net PnL (> 0.0 bps) and WR >= 75%, the Reigning Champion
           RETAINS its crown, and all losing challenger mutations are REJECTED!
        """
        leaderboard = self.update_metrics()
        champ_record = next((x for x in leaderboard if x["name"] == reigning_champ_name), leaderboard[0])

        # Filter challengers meeting the Absolute Peak Gate
        qualifying_challengers = [
            x for x in leaderboard
            if x["name"] != reigning_champ_name
            and x["net_pnl_bps"] > 0.0
            and x["win_rate_pct"] >= 75.0
            and x["net_pnl_bps"] > champ_record["net_pnl_bps"]
        ]

        if qualifying_challengers:
            # Sort by Net PnL, then Win Rate, then Trades
            qualifying_challengers.sort(key=lambda x: (x["net_pnl_bps"], x["win_rate_pct"], x["trades"]), reverse=True)
            winner = qualifying_challengers[0]
            log.info("🏆 NEW CHAMPION CROWNED: %s (%s) with +%.2f bps Net and %.1f%% WR!", winner["name"], winner["strategy_profile"], winner["net_pnl_bps"], winner["win_rate_pct"])
            return winner
        else:
            log.info("👑 REIGNING CHAMPION DEFENDED / PRESERVED: %s (%s) [No challenger satisfied absolute gate: net > 0.0 & WR >= 75.0%%]", champ_record["name"], champ_record["strategy_profile"])
            return champ_record
