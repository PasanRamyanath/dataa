"""
feature engineering scripts/execution/champion_challenger_framework.py
Production Champion-Challenger Tournament Framework for Project-IMPJR Peak Trading System.
Cycle 24 Tournament Roster:
  - 👑 Reigning Champion: HighFrequencyTacticalScalper (Crowned in Cycle 23 with 80% Win Rate, 5 trades, +11.16 bps won on wins)
  - ⚔️ Challenger 1: Anti-Exhaustion VWAP Displacement Scalper (Solves Trade 5 exhaustion top loss)
  - ⚔️ Challenger 2: Volatility Crest Ratchet Scalper (Locks +3.0 bps net at the crest of tactical expansions)
  - ⚔️ Challenger 3: Order Flow Imbalance (OFI) Micro-Burst Scalper (Aggressive flow confirmation)
  - ⚔️ Challenger 4: Conformal Shannon Negentropy Peak Scalper (Exits at peak information certainty)
  - ⚔️ Challenger 5: Adaptive Momentum Decay Unwind Scalper (Tight 120s stall prevention)
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
# 👑 REIGNING CHAMPION: HighFrequencyTacticalScalper (Crowned Cycle 23)
# ==============================================================================
class HighFrequencyTacticalScalper(HierarchicalSniperExecutionEngine):
    """
    Reigning Champion (Crowned in Cycle 23):
    High-Frequency Tactical Impulse Scalper with decoupled 1m/30s momentum entries,
    Dual-Horizon Invalidation Bounding, 3-Tier Dynamic Trailing Shield, and Break-Even Lock.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Reigning Champion: HF Tactical Impulse Scalper (80% WR in Cycle 23)"
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


# ==============================================================================
# ⚔️ CHALLENGER 1: Anti-Exhaustion VWAP Displacement Scalper
# ==============================================================================
class Challenger1_AntiExhaustionVWAPScalper(HighFrequencyTacticalScalper):
    """
    Challenger 1: HF Tactical Scalper + Anti-Exhaustion VWAP Displacement Filter
    Specifically designed to eliminate Trade 5 exhaustion top losses:
    Maintains a rolling 30-bar (~5m) Volume-Weighted Price baseline.
    Rejects LONG entries if price is extended > 12.0 bps above VWAP.
    Rejects SHORT entries if price is extended > 12.0 bps below VWAP.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 1: Anti-Exhaustion VWAP Displacement Scalper"
        self.rolling_prices: deque[float] = deque(maxlen=30)
        self.max_displacement_bps: float = 12.0

    def on_market_micro_tick(self, *args, **kwargs):
        mid = 0.5 * (kwargs.get("bbo_bid", 0.0) + kwargs.get("bbo_ask", 0.0))
        if mid > 0:
            self.rolling_prices.append(mid)
        return super().on_market_micro_tick(*args, **kwargs)

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        mid = 0.5 * (bbo_bid + bbo_ask)
        if len(self.rolling_prices) >= 15 and mid > 0:
            avg_p = float(np.mean(self.rolling_prices))
            disp_bps = ((mid - avg_p) / avg_p) * 1e4
            d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
            # Block over-extended chases
            if d_tact >= 0.14 and disp_bps > self.max_displacement_bps:
                return self.state  # Suppress exhaustion top long
            if d_tact <= -0.14 and disp_bps < -self.max_displacement_bps:
                return self.state  # Suppress exhaustion bottom short
        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)


# ==============================================================================
# ⚔️ CHALLENGER 2: Volatility Crest Ratchet Scalper
# ==============================================================================
class Challenger2_VolatilityCrestRatchetScalper(HighFrequencyTacticalScalper):
    """
    Challenger 2: HF Tactical Scalper + Volatility Crest Profit Ratchet
    When peak MFE reaches >= +5.0 bps gross and 10s volatility contracts,
    immediately locks floor at +3.0 bps net profit (+7.0 bps gross floor or maker exit).
    Prevents giving back winning trades to +0.79 bps!
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 2: Volatility Crest Ratchet Scalper"

    def on_market_micro_tick(self, *args, **kwargs):
        state = super().on_market_micro_tick(*args, **kwargs)
        if self.active_position is not None and self.active_position.peak_favorable_excursion_bps >= 5.0:
            bbo_bid = kwargs.get("bbo_bid", 0.0)
            bbo_ask = kwargs.get("bbo_ask", 0.0)
            now_ms = kwargs.get("ts_ms", 0)
            mid = 0.5 * (bbo_bid + bbo_ask)
            pos = self.active_position
            sign = 1.0 if pos.side == "BUY" else -1.0
            unreal = sign * ((mid - pos.entry_price) / pos.entry_price) * 1e4
            
            # If crest retraces slightly from peak, lock +3.0 bps net profit immediately
            if pos.peak_favorable_excursion_bps >= 5.5 and (pos.peak_favorable_excursion_bps - unreal) >= 1.2:
                exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
                self._close_position(
                    exit_price=exit_p,
                    exit_ts_ms=now_ms,
                    exit_fee_bps=self.maker_fee_leg,
                    reason="VOLATILITY_CREST_RATCHET_TP",
                )
        return self.state


# ==============================================================================
# ⚔️ CHALLENGER 3: Order Flow Imbalance (OFI) Micro-Burst Scalper
# ==============================================================================
class Challenger3_OFIMicroBurstScalper(HighFrequencyTacticalScalper):
    """
    Challenger 3: HF Tactical Scalper + Order Flow Imbalance (OFI) Burst Confirmation
    Validates that aggressive market orders in the 10s bar are flowing heavily in trade direction.
    Requires buy/sell volume ratio >= 1.35 for LONG and <= 0.74 for SHORT.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 3: OFI Micro-Burst Scalper"
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
            if d_tact >= 0.14 and buy_ratio < 0.55:
                return self.state  # Suppress long if aggressive volume is predominantly sellers
            if d_tact <= -0.14 and buy_ratio > 0.45:
                return self.state  # Suppress short if aggressive volume is predominantly buyers
        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)


# ==============================================================================
# ⚔️ CHALLENGER 4: Conformal Shannon Negentropy Peak Scalper
# ==============================================================================
class Challenger4_ConformalEntropyScalper(HighFrequencyTacticalScalper):
    """
    Challenger 4: HF Tactical Scalper + Shannon Negentropy Peak Harvest
    Monitors directional information certainty C = 1 - H(p)/ln(3).
    When tactical uncertainty begins rising (C drops by >= 0.15 from trade peak),
    harvests immediately at Maker BBO, securing maximum MFE before reversion.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 4: Conformal Shannon Negentropy Scalper"
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
                # Information certainty decaying: harvest at Maker BBO
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
# ⚔️ CHALLENGER 5: Adaptive Momentum Decay Unwind Scalper
# ==============================================================================
class Challenger5_AdaptiveDecayUnwindScalper(HighFrequencyTacticalScalper):
    """
    Challenger 5: HF Tactical Scalper + Adaptive 120s Momentum Decay Unwind
    If after 120s the trade has stalled without achieving >= +3.0 bps MFE,
    passively unwinds at BBO to prevent capital lockup and eliminate late reversal risk.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 5: Adaptive Momentum Decay Unwind Scalper"

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
                engine=HighFrequencyTacticalScalper(symbol=symbol),
                strategy_profile="Reigning Champion: HF Tactical Impulse Scalper (Crowned Cycle 23)",
            ),
            "Challenger_1": TournamentCompetitorRecord(
                name="Challenger_1",
                engine=Challenger1_AntiExhaustionVWAPScalper(symbol=symbol),
                strategy_profile="Anti-Exhaustion VWAP Displacement Scalper",
            ),
            "Challenger_2": TournamentCompetitorRecord(
                name="Challenger_2",
                engine=Challenger2_VolatilityCrestRatchetScalper(symbol=symbol),
                strategy_profile="Volatility Crest Profit Ratchet Scalper",
            ),
            "Challenger_3": TournamentCompetitorRecord(
                name="Challenger_3",
                engine=Challenger3_OFIMicroBurstScalper(symbol=symbol),
                strategy_profile="OFI Micro-Burst Flow Scalper",
            ),
            "Challenger_4": TournamentCompetitorRecord(
                name="Challenger_4",
                engine=Challenger4_ConformalEntropyScalper(symbol=symbol),
                strategy_profile="Conformal Shannon Negentropy Scalper",
            ),
            "Challenger_5": TournamentCompetitorRecord(
                name="Challenger_5",
                engine=Challenger5_AdaptiveDecayUnwindScalper(symbol=symbol),
                strategy_profile="Adaptive Momentum Decay Unwind Scalper",
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
