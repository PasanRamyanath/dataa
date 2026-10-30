"""
scratch/test_cycle37_challengers_simulation.py
Simulates the Cycle 37 lineup on archived Cycle 35 and Cycle 36 data
to verify that ALL competitors maintain 100% win-rate safety and positive net PnL.
"""
import sys
sys.path.insert(0, "feature engineering scripts")
import json
import copy
from pathlib import Path
from execution.champion_challenger_framework import (
    TournamentCompetitorRecord,
    Challenger5_ConformalConfidenceFilterScalper,
)
from fusion.hierarchical_fusion_v3 import HierarchicalDecisionOutput

# ==============================================================================
# ⚔️ CHALLENGER 1: Conformal Bidirectional Dual Scalper
# ==============================================================================
class Challenger1_ConformalBidirectionalDualScalper(Challenger5_ConformalConfidenceFilterScalper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 1: Conformal Bidirectional Dual Scalper"

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        now_ms = decision.ts_ms
        mid = 0.5 * (bbo_bid + bbo_ask)
        self.recent_mid_prices.append((now_ms, mid))
        self.rolling_15m_prices.append(mid)
        d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
        d_micro = decision.micro_probs[2] - decision.micro_probs[0]
        d_meso = decision.meso_probs[2] - decision.meso_probs[0]

        conf_tact = max(decision.tactical_probs)
        conf_meso = max(decision.meso_probs)
        if conf_tact < 0.45 or conf_meso < 0.45:
            return self.state

        if self.state == "IDLE" and self.ticks_seen >= 10:
            tact_dir = 2 if d_tact >= 0.11 else (0 if d_tact <= -0.11 else 1)
            # Bidirectional: Allows short if tactical is confident down and micro agrees, even if meso is neutral
            tact_confirmed = (tact_dir in (0, 2)) and (
                (tact_dir == 2 and d_micro > 0.015 and d_meso >= 0.02) or
                (tact_dir == 0 and d_micro < -0.015 and d_meso <= 0.01)
            )

            cand_dir = tact_dir if tact_confirmed else None
            if cand_dir is None and decision.primary_direction in (0, 2) and decision.expected_net_edge_bps >= self.min_net_edge_bps:
                if decision.primary_direction == 2 and d_meso >= 0.02:
                    cand_dir = 2
                elif decision.primary_direction == 0 and d_meso <= 0.01:
                    cand_dir = 0

            if cand_dir in (0, 2):
                if self._is_range_boundary_violation(cand_dir, mid):
                    return self.state
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

        return super(Challenger5_ConformalConfidenceFilterScalper, self).on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)

# ==============================================================================
# ⚔️ CHALLENGER 2: Conformal Neural Saturation Scalper
# ==============================================================================
class Challenger2_ConformalNeuralSaturationScalper(Challenger5_ConformalConfidenceFilterScalper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 2: Conformal Neural Saturation Scalper"

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

        # Active Neural Peak Harvest: once in profit >= 6.0 bps gross (+2.0 bps net), if momentum saturates, exit immediately at Maker
        if unrealized_bps >= 6.00 and s_decay <= 0.00:
            maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
            self._close_position(
                exit_price=maker_exit_p,
                exit_ts_ms=now_ms,
                exit_fee_bps=self.maker_fee_leg,
                reason="CONFORMAL_NEURAL_SATURATION_HARVEST",
            )
            return

        super()._manage_active_swing_position(decision, bbo_bid, bbo_ask, now_ms)

# ==============================================================================
# ⚔️ CHALLENGER 3: Conformal Dynamic Pullback Limit Scalper
# ==============================================================================
class Challenger3_ConformalDynamicPullbackLimitScalper(Challenger5_ConformalConfidenceFilterScalper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 3: Conformal Dynamic Pullback Limit Scalper"

    def _place_sniper_maker_entry(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, now_ms, intra_bar_open):
        dir_val = decision.primary_direction
        pos_side = "BUY" if dir_val == 2 else "SELL"
        # Rest limit 1 tick deeper ($0.50) on pullback to capture extra spread margin
        if pos_side == "BUY":
            limit_p = bbo_bid - 0.50
        else:
            limit_p = bbo_ask + 0.50

        qty = round((equity_usdt * 0.10) / limit_p, 3)
        self.resting_entry_order = {
            "order_id": f"SNIPER_CONF_PULLBACK_{pos_side}_{now_ms}",
            "side": pos_side,
            "price": limit_p,
            "quantity": qty,
            "ts_ms": now_ms,
            "expected_dir": dir_val,
            "source": "CONFORMAL_PULLBACK_MAKER",
        }
        self.state = "PLACING_MAKER_ENTRY"

# ==============================================================================
# ⚔️ CHALLENGER 4: Conformal Asymmetric Ratchet Scalper
# ==============================================================================
class Challenger4_ConformalAsymmetricRatchetScalper(Challenger5_ConformalConfidenceFilterScalper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 4: Conformal Asymmetric Ratchet Scalper"

    def on_market_micro_tick(self, ts_ms: int, bbo_bid: float, bbo_ask: float, bbo_bid_qty: float, bbo_ask_qty: float, *args, **kwargs):
        if self.state in ("FILLED_SWING_HOLDING", "PASSIVE_MAKER_UNWIND") and self.active_position is not None:
            pos = self.active_position
            sign = 1.0 if pos.side == "BUY" else -1.0
            mid = 0.5 * (bbo_bid + bbo_ask)
            unrealized_bps = sign * ((mid - pos.entry_price) / pos.entry_price) * 1e4
            if unrealized_bps > pos.peak_favorable_excursion_bps:
                pos.peak_favorable_excursion_bps = unrealized_bps

            # Stepped ratchet locks for high-conviction conformal trades
            mfe = pos.peak_favorable_excursion_bps
            floor_gross = None
            if mfe >= 15.0:
                floor_gross = 12.0  # +8.0 bps net
            elif mfe >= 10.0:
                floor_gross = 8.0   # +4.0 bps net
            elif mfe >= 7.0:
                floor_gross = 6.0   # +2.0 bps net (locked ahead of break-even)

            if floor_gross is not None and unrealized_bps <= floor_gross:
                exit_p = pos.entry_price * (1.0 + sign * floor_gross * 1e-4)
                self._close_position(
                    exit_price=exit_p,
                    exit_ts_ms=ts_ms,
                    exit_fee_bps=self.maker_fee_leg,
                    reason="CONFORMAL_ASYMMETRIC_RATCHET_TP",
                )
                return self.state

        return super().on_market_micro_tick(ts_ms=ts_ms, bbo_bid=bbo_bid, bbo_ask=bbo_ask, bbo_bid_qty=bbo_bid_qty, bbo_ask_qty=bbo_ask_qty, *args, **kwargs)

# ==============================================================================
# ⚔️ CHALLENGER 5: Conformal Volatility Expansion Scalper
# ==============================================================================
class Challenger5_ConformalVolatilityExpansionScalper(Challenger5_ConformalConfidenceFilterScalper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 5: Conformal Volatility Expansion Scalper"

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        # Calculate recent 30-bar range
        if len(self.recent_mid_prices) >= 10:
            prices = [p[1] for p in list(self.recent_mid_prices)[-30:]]
            rng = max(prices) - min(prices)
            # Expand take profit during volatile expansion
            if rng >= 30.0:
                self.take_profit_bps = 35.0
            else:
                self.take_profit_bps = 15.0
        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)


def simulate_lineup(cycle_num: int):
    ledger_path = Path(f"scratch/archive_cycle{cycle_num}/live_hierarchical_session_1h_ledger.jsonl")
    if not ledger_path.exists():
        print(f"Cycle {cycle_num} ledger not found: {ledger_path}")
        return

    bars = []
    with open(ledger_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                bars.append(json.loads(line))

    competitors = {
        "Champion": Challenger5_ConformalConfidenceFilterScalper(symbol="BTCUSDT"),
        "Challenger_1": Challenger1_ConformalBidirectionalDualScalper(symbol="BTCUSDT"),
        "Challenger_2": Challenger2_ConformalNeuralSaturationScalper(symbol="BTCUSDT"),
        "Challenger_3": Challenger3_ConformalDynamicPullbackLimitScalper(symbol="BTCUSDT"),
        "Challenger_4": Challenger4_ConformalAsymmetricRatchetScalper(symbol="BTCUSDT"),
        "Challenger_5": Challenger5_ConformalVolatilityExpansionScalper(symbol="BTCUSDT"),
    }

    for b in bars:
        dec = HierarchicalDecisionOutput(
            macro_probs=b["macro_probs"],
            meso_probs=b["meso_probs"],
            tactical_probs=b["tactical_probs"],
            micro_probs=b["micro_probs"],
            fused_swing_probs=b.get("fused_swing_probs", b["tactical_probs"]),
            primary_direction=b["primary_direction"],
            cross_horizon_alignment=b.get("cross_horizon_alignment", 1.0),
            expected_gross_move_bps=b.get("expected_gross_move_bps", 12.0),
            expected_net_edge_bps=b.get("expected_net_edge_bps", 6.0),
            meta_act_probability=b.get("meta_act_probability", 0.8),
            micro_sniper_trigger_prob=b.get("micro_sniper_trigger_prob", 0.7),
            target_holding_seconds=b.get("target_holding_seconds", 300),
            take_profit_bps=b.get("take_profit_bps", 30.0),
            stop_loss_bps=b.get("stop_loss_bps", 14.5),
            kelly_position_fraction=b.get("kelly_position_fraction", 0.5),
            execution_mode="PASSIVE_MAKER_SNIPER",
            ts_ms=b["ts_ms"],
        )

        bbo_bid = b["bbo_bid"]
        bbo_ask = b["bbo_ask"]
        bbo_bid_qty = b.get("bbo_bid_qty", 1.5)
        bbo_ask_qty = b.get("bbo_ask_qty", 1.5)
        bar_high = b.get("bar_high", bbo_ask)
        bar_low = b.get("bar_low", bbo_bid)

        for eng in competitors.values():
            eng.on_market_micro_tick(
                ts_ms=b["ts_ms"],
                bbo_bid=bbo_bid,
                bbo_ask=bbo_ask,
                bbo_bid_qty=bbo_bid_qty,
                bbo_ask_qty=bbo_ask_qty,
                bar_high=bar_high,
                bar_low=bar_low,
                current_decision=dec,
            )
            eng.on_hierarchical_decision(
                decision=dec,
                bbo_bid=bbo_bid,
                bbo_ask=bbo_ask,
                bbo_bid_qty=bbo_bid_qty,
                bbo_ask_qty=bbo_ask_qty,
                equity_usdt=10000.0,
                intra_bar_open=b.get("intra_bar_open", bbo_bid),
            )

    print("=" * 90)
    print(f"CYCLE 37 LINEUP SIMULATION RESULTS FOR CYCLE {cycle_num}")
    print("=" * 90)
    for name, eng in competitors.items():
        tr = eng.closed_trades
        n_tr = len(tr)
        wins = sum(1 for t in tr if t.net_return_bps > 0)
        wr = (wins / n_tr * 100) if n_tr > 0 else 0.0
        tot_net = sum(t.net_return_bps for t in tr)
        print(f"  {name:15s} | Trades: {n_tr:2d} | Wins: {wins:2d} | WR: {wr:5.1f}% | Net PnL: {tot_net:+6.2f} bps | Profile: {eng.strategy_profile}")
        for i, t in enumerate(tr, 1):
            print(f"      T{i}: {t.side} @ {t.entry_price:.2f} -> {t.exit_price:.2f} | Net: {t.net_return_bps:+6.2f} bps | MFE: {t.peak_mfe_bps:+5.2f}b | Hold: {t.holding_duration_sec:.0f}s | Reason: {t.exit_reason}")

if __name__ == "__main__":
    simulate_lineup(35)
    simulate_lineup(36)
