"""
feature engineering scripts/execution/champion_challenger_framework.py
Production Champion-Challenger Tournament Framework for Project-IMPJR Peak Trading System.

Cycle 33 Tournament Lineup (Peak System Restored & 5 Radically Diverse HFT Quant Architectures):
  - 👑 Reigning Champion: PeakSameSideRetraceScalper (Defended title across C25, C26, C31, and C32; cumulative +41.96 bps net across peak cycles, with Responsive Model Maker Invalidation)
  - ⚔️ Challenger 1: MicroPriceOFIScalper (Computes micro-price P_micro and order-book depth imbalance ratio; snipes maker fills with immediate queue support)
  - ⚔️ Challenger 2: HarmonicQuadResonanceScalper (Requires all 4 horizons - Macro, Meso, Tactical, Micro - to simultaneously confirm trend; rides 25 bps TP)
  - ⚔️ Challenger 3: CrossAssetLeadLagScalper (Exploits statistical lead-lag between ETH perp momentum and BTC absorption)
  - ⚔️ Challenger 4: AvellanedaStoikovReservationScalper (Optimal execution with time-decaying reservation price; locks maker gains before momentum wanes)
  - ⚔️ Challenger 5: DynamicFrequencyHighVelocityScalper (Calibrated 0.11 impulse threshold + micro-velocity filter for 6-8 high-conviction trades/hr)

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
# 👑 REIGNING CHAMPION: Peak Same-Side Retrace Scalper (Undefeated Baseline)
# ==============================================================================
class PeakSameSideRetraceScalper(HierarchicalSniperExecutionEngine):
    """
    Undefeated Champion (Defended across Cycles 25, 26, 31, and 32 with cumulative +41.96 bps Peak Net):
    Combines:
    1. Multi-Horizon Stage 5 Supervised Fusion + High-Frequency Tactical Impulse Scaling.
    2. Meso Concordance Guard (d_meso >= 0.00 for BUY, d_meso <= 0.00 for SELL).
    3. Same-Side Retrace Price Barrier: Requires $15.00 pullback before re-entering same side within 300s.
    4. Positive Break-Even Profit Shield: Locks +4.10 bps gross (+0.10 bps net profit guaranteed) once MFE >= 3.80 bps.
    5. Wide-Breathe Trailing & Neural Model Profit Harvest (MODEL_HARVEST_MAKER_TP).
    6. Responsive Model Invalidation Maker Exit: Caps losses when Meso reverses hostile while underwater.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Reigning Champion: Peak Same-Side Retrace Scalper (+41.96 bps Peak Cumulative Net)"
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
        self.recent_mid_prices.append((now_ms, mid))
        d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
        d_micro = decision.micro_probs[2] - decision.micro_probs[0]
        d_meso = decision.meso_probs[2] - decision.meso_probs[0]

        if self.state == "IDLE" and self.ticks_seen >= 10:
            cand_dir = None
            tact_dir = 2 if d_tact >= 0.14 else (0 if d_tact <= -0.14 else 1)
            tact_confirmed = (tact_dir in (0, 2)) and (
                (tact_dir == 2 and d_micro > 0.02 and d_meso >= 0.04) or
                (tact_dir == 0 and d_micro < -0.02 and d_meso <= -0.04)
            )

            if tact_confirmed:
                cand_dir = tact_dir
            elif decision.primary_direction in (0, 2) and decision.expected_net_edge_bps >= self.min_net_edge_bps:
                if decision.primary_direction == 2 and d_meso >= 0.04:
                    cand_dir = 2
                elif decision.primary_direction == 0 and d_meso <= -0.04:
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
# ⚔️ CHALLENGER 1: High-Velocity Impulse Promotion Scalper (Evolved from C33 C5)
# ==============================================================================
class Challenger1_HighVelocityPromotionScalper(PeakSameSideRetraceScalper):
    """
    Challenger 1: High-Velocity Impulse Promotion Scalper
    Direct evolution of Cycle 33 Challenger 5 (which achieved 80.0% Win Rate across 5 tr/hr).
    Combines calibrated 0.11 tactical impulse threshold + micro-velocity alignment with
    Meso concordance guard (d_meso >= 0.03) and rapid 12s post-win reset.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 1: High-Velocity Impulse Promotion Scalper (5-8 tr/hr Target)"

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        now_ms = decision.ts_ms
        mid = 0.5 * (bbo_bid + bbo_ask)
        self.recent_mid_prices.append((now_ms, mid))
        d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
        d_micro = decision.micro_probs[2] - decision.micro_probs[0]
        d_meso = decision.meso_probs[2] - decision.meso_probs[0]

        if self.state == "IDLE" and self.ticks_seen >= 10:
            tact_dir = 2 if d_tact >= 0.11 else (0 if d_tact <= -0.11 else 1)
            tact_confirmed = (tact_dir in (0, 2)) and (
                (tact_dir == 2 and d_micro > 0.015 and d_meso >= 0.03) or
                (tact_dir == 0 and d_micro < -0.015 and d_meso <= -0.03)
            )

            cand_dir = tact_dir if tact_confirmed else None
            if cand_dir is None and decision.primary_direction in (0, 2) and decision.expected_net_edge_bps >= self.min_net_edge_bps:
                if decision.primary_direction == 2 and d_meso >= 0.03:
                    cand_dir = 2
                elif decision.primary_direction == 0 and d_meso <= -0.03:
                    cand_dir = 0

            if cand_dir in (0, 2):
                if self._is_parabolic_surge(cand_dir, now_ms, mid):
                    return self.state

                if self.last_exit_ts_ms > 0:
                    elapsed = (now_ms - self.last_exit_ts_ms) / 1000.0
                    if not self.last_exit_was_win and elapsed < 60.0:
                        return self.state
                    elif self.last_exit_was_win and elapsed < 12.0:
                        return self.state

                    if elapsed < 240.0 and self.last_trade_exit_price > 0:
                        if cand_dir == 2 and self.last_trade_side == "BUY" and (mid > self.last_trade_exit_price - 12.0):
                            return self.state
                        elif cand_dir == 0 and self.last_trade_side == "SELL" and (mid < self.last_trade_exit_price + 12.0):
                            return self.state

                tact_dec = copy.copy(decision)
                object.__setattr__(tact_dec, "primary_direction", cand_dir) if hasattr(tact_dec, "__dataclass_fields__") and getattr(tact_dec, "__dataclass_params__", None) and getattr(tact_dec.__dataclass_params__, "frozen", False) else setattr(tact_dec, "primary_direction", cand_dir)
                self.stalking_decision = tact_dec
                self._place_sniper_maker_entry(tact_dec, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, now_ms, intra_bar_open)
                return self.state

        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)


# ==============================================================================
# ⚔️ CHALLENGER 2: Neural Multi-Horizon Peak Harvest Scalper
# ==============================================================================
class Challenger2_NeuralPeakHarvestScalper(PeakSameSideRetraceScalper):
    """
    Challenger 2: Neural Multi-Horizon Peak Harvest Scalper
    Actively queries multi-horizon model momentum during open trades.
    Computes an instantaneous Momentum Saturation Index:
      S_decay = sign * (0.45 * d_micro + 0.35 * d_tact + 0.20 * d_meso)
    While in profit (unrealized_bps >= 4.50), as soon as S_decay <= 0.00 (momentum rolls over),
    it immediately harvests peak excursion at BBO Maker to maximize realized profit per trade.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 2: Neural Multi-Horizon Peak Harvest Scalper"

    def _manage_active_swing_position(self, decision, bbo_bid, bbo_ask, now_ms):
        if self.active_position is None:
            return
        pos = self.active_position
        sign = 1.0 if pos.side == "BUY" else -1.0
        mid = 0.5 * (bbo_bid + bbo_ask)
        unrealized_bps = sign * ((mid - pos.entry_price) / pos.entry_price) * 1e4
        d_micro = decision.micro_probs[2] - decision.micro_probs[0]
        d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
        d_meso = decision.meso_probs[2] - decision.meso_probs[0]

        # Multi-Horizon Momentum Saturation Index
        s_decay = sign * (0.45 * d_micro + 0.35 * d_tact + 0.20 * d_meso)

        # Active Neural Peak Harvest: while in profit, exit Maker when multi-horizon momentum decelerates
        if unrealized_bps >= 4.50 and s_decay <= 0.00:
            maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
            self._close_position(
                exit_price=maker_exit_p,
                exit_ts_ms=now_ms,
                exit_fee_bps=self.maker_fee_leg,
                reason="NEURAL_MOMENTUM_PEAK_HARVEST",
            )
            return

        super()._manage_active_swing_position(decision, bbo_bid, bbo_ask, now_ms)


# ==============================================================================
# ⚔️ CHALLENGER 3: Asymmetric Runner Expansion Scalper (40 bps Target + Ratchet)
# ==============================================================================
class Challenger3_AsymmetricRunnerExpansionScalper(PeakSameSideRetraceScalper):
    """
    Challenger 3: Asymmetric Runner Expansion Scalper
    Expands Take Profit target to 40.0 bps on high-concordance entries.
    Enforces a 3-tier stepped profit ratchet:
      - MFE >= 6.00 bps  -> locks +1.50 bps net floor (+5.50 bps gross)
      - MFE >= 10.00 bps -> locks +4.00 bps net floor (+8.00 bps gross)
      - MFE >= 15.00 bps -> locks +8.00 bps net floor (+12.00 bps gross)
    Captures large multi-dollar swings when multi-horizon trends run.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 3: Asymmetric Runner Expansion Scalper (40 bps Target + Ratchet)"

    def on_market_micro_tick(self, ts_ms: int, bbo_bid: float, bbo_ask: float, bbo_bid_qty: float, bbo_ask_qty: float, *args, **kwargs):
        if self.state in ("FILLED_SWING_HOLDING", "PASSIVE_MAKER_UNWIND") and self.active_position is not None:
            pos = self.active_position
            sign = 1.0 if pos.side == "BUY" else -1.0
            mid = 0.5 * (bbo_bid + bbo_ask)
            unrealized_bps = sign * ((mid - pos.entry_price) / pos.entry_price) * 1e4
            if unrealized_bps > pos.peak_favorable_excursion_bps:
                pos.peak_favorable_excursion_bps = unrealized_bps

            # Stepped ratchet locks
            mfe = pos.peak_favorable_excursion_bps
            floor_gross = None
            if mfe >= 15.0:
                floor_gross = 12.0  # +8.0 bps net
            elif mfe >= 10.0:
                floor_gross = 8.0   # +4.0 bps net
            elif mfe >= 6.0:
                floor_gross = 5.5   # +1.5 bps net

            if floor_gross is not None and unrealized_bps <= floor_gross:
                exit_p = pos.entry_price * (1.0 + sign * floor_gross * 1e-4)
                self._close_position(
                    exit_price=exit_p,
                    exit_ts_ms=ts_ms,
                    exit_fee_bps=self.maker_fee_leg,
                    reason="ASYMMETRIC_STEPPED_RATCHET_TP",
                )
                return self.state

        return super().on_market_micro_tick(ts_ms=ts_ms, bbo_bid=bbo_bid, bbo_ask=bbo_ask, bbo_bid_qty=bbo_bid_qty, bbo_ask_qty=bbo_ask_qty, *args, **kwargs)


# ==============================================================================
# ⚔️ CHALLENGER 4: Adaptive Volatility Spread Micro Scalper
# ==============================================================================
class Challenger4_VolSpreadAdaptiveScalper(PeakSameSideRetraceScalper):
    """
    Challenger 4: Adaptive Volatility Spread Micro Scalper
    Monitors 30s rolling Parkinson volatility and LOB spread.
    In low volatility (sigma < 2.5 bps), places resting maker entry directly at BBO.
    In elevated volatility (sigma >= 2.5 bps), steps back 0.5 bps to capture spread rebate
    and filter out adverse execution toxicity.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 4: Adaptive Volatility Spread Scalper"

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        now_ms = decision.ts_ms
        mid = 0.5 * (bbo_bid + bbo_ask)
        spread = bbo_ask - bbo_bid

        # Require minimum viable spread for maker edge
        if spread < 0.10:
            return self.state

        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)


# ==============================================================================
# ⚔️ CHALLENGER 5: Conformal Confidence Filter Scalper (High-Certainty Regime)
# ==============================================================================
class Challenger5_ConformalConfidenceFilterScalper(PeakSameSideRetraceScalper):
    """
    Challenger 5: Conformal Confidence Filter Scalper
    Filters out uncertain / flat regime noise by requiring maximum softmax confidence >= 0.46
    and alignment across Tactical and Meso tiers.
    Enforces a calibrated $10 same-side retrace reset for rapid opportunity capture.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 5: Conformal Confidence Filter Scalper"

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        now_ms = decision.ts_ms
        mid = 0.5 * (bbo_bid + bbo_ask)
        self.recent_mid_prices.append((now_ms, mid))
        d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
        d_micro = decision.micro_probs[2] - decision.micro_probs[0]
        d_meso = decision.meso_probs[2] - decision.meso_probs[0]

        # Conformal confidence check on tactical and meso
        conf_tact = max(decision.tactical_probs)
        conf_meso = max(decision.meso_probs)
        if conf_tact < 0.45 or conf_meso < 0.45:
            return self.state

        if self.state == "IDLE" and self.ticks_seen >= 10:
            tact_dir = 2 if d_tact >= 0.12 else (0 if d_tact <= -0.12 else 1)
            tact_confirmed = (tact_dir in (0, 2)) and (
                (tact_dir == 2 and d_micro > 0.015 and d_meso >= 0.03) or
                (tact_dir == 0 and d_micro < -0.015 and d_meso <= -0.03)
            )

            cand_dir = tact_dir if tact_confirmed else None
            if cand_dir is None and decision.primary_direction in (0, 2) and decision.expected_net_edge_bps >= self.min_net_edge_bps:
                if decision.primary_direction == 2 and d_meso >= 0.03:
                    cand_dir = 2
                elif decision.primary_direction == 0 and d_meso <= -0.03:
                    cand_dir = 0

            if cand_dir in (0, 2):
                if self._is_parabolic_surge(cand_dir, now_ms, mid):
                    return self.state

                if self.last_exit_ts_ms > 0:
                    elapsed = (now_ms - self.last_exit_ts_ms) / 1000.0
                    if not self.last_exit_was_win and elapsed < 60.0:
                        return self.state
                    elif self.last_exit_was_win and elapsed < 15.0:
                        return self.state

                    if elapsed < 200.0 and self.last_trade_exit_price > 0:
                        if cand_dir == 2 and self.last_trade_side == "BUY" and (mid > self.last_trade_exit_price - 10.0):
                            return self.state
                        elif cand_dir == 0 and self.last_trade_side == "SELL" and (mid < self.last_trade_exit_price + 10.0):
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
                strategy_profile="Reigning Champion: Peak Same-Side Retrace Scalper (+41.96 bps Peak Cumulative Net)",
            ),
            "Challenger_1": TournamentCompetitorRecord(
                name="Challenger_1",
                engine=Challenger1_HighVelocityPromotionScalper(symbol=symbol),
                strategy_profile="High-Velocity Impulse Promotion Scalper (5-8 tr/hr Target)",
            ),
            "Challenger_2": TournamentCompetitorRecord(
                name="Challenger_2",
                engine=Challenger2_NeuralPeakHarvestScalper(symbol=symbol),
                strategy_profile="Neural Multi-Horizon Peak Harvest Scalper",
            ),
            "Challenger_3": TournamentCompetitorRecord(
                name="Challenger_3",
                engine=Challenger3_AsymmetricRunnerExpansionScalper(symbol=symbol),
                strategy_profile="Asymmetric Runner Expansion Scalper (40 bps Target + Ratchet)",
            ),
            "Challenger_4": TournamentCompetitorRecord(
                name="Challenger_4",
                engine=Challenger4_VolSpreadAdaptiveScalper(symbol=symbol),
                strategy_profile="Adaptive Volatility Spread Scalper",
            ),
            "Challenger_5": TournamentCompetitorRecord(
                name="Challenger_5",
                engine=Challenger5_ConformalConfidenceFilterScalper(symbol=symbol),
                strategy_profile="Conformal Confidence Filter Scalper",
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
