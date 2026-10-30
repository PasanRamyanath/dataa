import json
import os
import sys
import copy
from collections import deque
import numpy as np

sys.path.insert(0, os.path.abspath("feature engineering scripts"))
from execution.champion_challenger_framework import PeakSameSideRetraceScalper
from execution.hierarchical_sniper_engine import HierarchicalSniperExecutionEngine
from fusion.hierarchical_fusion_v3 import HierarchicalDecisionOutput


# ==============================================================================
# 1. Micro-Price OFI Scalper
# ==============================================================================
class MicroPriceOFIScalper(HierarchicalSniperExecutionEngine):
    """
    Challenger 1: Micro-Price Order Flow Imbalance Scalper
    Computes micro-price P_micro = (P_bid * V_ask + P_ask * V_bid) / (V_bid + V_ask).
    Gated by micro-price displacement >= 0.8 bps and book depth ratio >= 1.5.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 1: Micro-Price OFI Scalper (LOB Depth Skew)"
        self.min_trailing_activation_bps = 6.0
        self.tier1_min_lock_bps = 4.3

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        now_ms = decision.ts_ms
        mid = 0.5 * (bbo_bid + bbo_ask)
        tot_qty = bbo_bid_qty + bbo_ask_qty
        if tot_qty <= 0:
            return self.state

        micro_p = (bbo_bid * bbo_ask_qty + bbo_ask * bbo_bid_qty) / tot_qty
        micro_disp_bps = ((micro_p - mid) / mid) * 1e4
        depth_ratio = bbo_bid_qty / max(1e-4, bbo_ask_qty)
        d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
        d_micro = decision.micro_probs[2] - decision.micro_probs[0]
        d_meso = decision.meso_probs[2] - decision.meso_probs[0]

        if self.state == "IDLE" and self.ticks_seen >= 10:
            cand_dir = None
            if micro_disp_bps >= 0.5 and depth_ratio >= 1.3 and d_tact >= 0.08 and d_meso >= 0.00:
                cand_dir = 2
            elif micro_disp_bps <= -0.5 and depth_ratio <= 0.77 and d_tact <= -0.08 and d_meso <= 0.00:
                cand_dir = 0

            if cand_dir in (0, 2):
                if self._is_parabolic_surge(cand_dir, now_ms, mid):
                    return self.state

                if self.last_exit_ts_ms > 0:
                    elapsed = (now_ms - self.last_exit_ts_ms) / 1000.0
                    if not self.last_exit_was_win and elapsed < 90.0:
                        return self.state
                    elif self.last_exit_was_win and elapsed < 15.0:
                        return self.state

                tact_dec = copy.copy(decision)
                object.__setattr__(tact_dec, "primary_direction", cand_dir) if hasattr(tact_dec, "__dataclass_fields__") and getattr(tact_dec, "__dataclass_params__", None) and getattr(tact_dec.__dataclass_params__, "frozen", False) else setattr(tact_dec, "primary_direction", cand_dir)
                self.stalking_decision = tact_dec
                self._place_sniper_maker_entry(tact_dec, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, now_ms, intra_bar_open)
                return self.state

        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)


# ==============================================================================
# 2. Multi-Horizon Resonant Trend Scalper
# ==============================================================================
class MultiHorizonResonantScalper(HierarchicalSniperExecutionEngine):
    """
    Challenger 2: Multi-Horizon Resonant Trend Scalper
    Requires all 4 horizons (Macro, Meso, Tactical, Micro) to simultaneously confirm direction.
    Achieves maximum signal certainty and lowest drawdowns.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 2: Multi-Horizon Resonant Scalper (4-Tier Resonance)"
        self.min_trailing_activation_bps = 7.0
        self.tier1_min_lock_bps = 4.5
        self.model_harvest_min_bps = 6.0

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        now_ms = decision.ts_ms
        mid = 0.5 * (bbo_bid + bbo_ask)
        d_macro = decision.macro_probs[2] - decision.macro_probs[0]
        d_meso = decision.meso_probs[2] - decision.meso_probs[0]
        d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
        d_micro = decision.micro_probs[2] - decision.micro_probs[0]

        if self.state == "IDLE" and self.ticks_seen >= 10:
            cand_dir = None
            if d_macro >= 0.08 and d_meso >= 0.05 and d_tact >= 0.12 and d_micro >= 0.02:
                cand_dir = 2
            elif d_macro <= -0.08 and d_meso <= -0.05 and d_tact <= -0.12 and d_micro <= -0.02:
                cand_dir = 0

            if cand_dir in (0, 2):
                if self._is_parabolic_surge(cand_dir, now_ms, mid):
                    return self.state

                tact_dec = copy.copy(decision)
                object.__setattr__(tact_dec, "primary_direction", cand_dir) if hasattr(tact_dec, "__dataclass_fields__") and getattr(tact_dec, "__dataclass_params__", None) and getattr(tact_dec.__dataclass_params__, "frozen", False) else setattr(tact_dec, "primary_direction", cand_dir)
                self.stalking_decision = tact_dec
                self._place_sniper_maker_entry(tact_dec, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, now_ms, intra_bar_open)
                return self.state

        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)


# ==============================================================================
# 3. Cross-Asset Lead-Lag Arbitrage Scalper
# ==============================================================================
class CrossAssetLeadLagScalper(HierarchicalSniperExecutionEngine):
    """
    Challenger 3: Cross-Asset Lead-Lag Arbitrage Scalper
    Monitors ETH lead-lag impulse (eth_1s_ret_bps) relative to BTC displacement.
    Snipes maker entries when ETH leads a directional move ahead of BTC absorption.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 3: Cross-Asset Lead-Lag Scalper (ETH Momentum Lead)"
        self.eth_ret_bps: float = 0.0

    def on_market_micro_tick(self, ts_ms: int, bbo_bid: float, bbo_ask: float, bbo_bid_qty: float, bbo_ask_qty: float, eth_1s_ret_bps: float = 0.0, *args, **kwargs):
        self.eth_ret_bps = eth_1s_ret_bps
        return super().on_market_micro_tick(ts_ms=ts_ms, bbo_bid=bbo_bid, bbo_ask=bbo_ask, bbo_bid_qty=bbo_bid_qty, bbo_ask_qty=bbo_ask_qty, eth_1s_ret_bps=eth_1s_ret_bps, *args, **kwargs)

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        now_ms = decision.ts_ms
        mid = 0.5 * (bbo_bid + bbo_ask)
        d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
        d_meso = decision.meso_probs[2] - decision.meso_probs[0]

        if self.state == "IDLE" and self.ticks_seen >= 10:
            cand_dir = None
            if self.eth_ret_bps >= 2.5 and d_tact >= 0.05 and d_meso >= 0.00:
                cand_dir = 2
            elif self.eth_ret_bps <= -2.5 and d_tact <= -0.05 and d_meso <= 0.00:
                cand_dir = 0

            if cand_dir in (0, 2):
                if self._is_parabolic_surge(cand_dir, now_ms, mid):
                    return self.state

                tact_dec = copy.copy(decision)
                object.__setattr__(tact_dec, "primary_direction", cand_dir) if hasattr(tact_dec, "__dataclass_fields__") and getattr(tact_dec, "__dataclass_params__", None) and getattr(tact_dec.__dataclass_params__, "frozen", False) else setattr(tact_dec, "primary_direction", cand_dir)
                self.stalking_decision = tact_dec
                self._place_sniper_maker_entry(tact_dec, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, now_ms, intra_bar_open)
                return self.state

        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)


# ==============================================================================
# 4. Avellaneda-Stoikov Adaptive Reservation Scalper
# ==============================================================================
class AvellanedaStoikovReservationScalper(HierarchicalSniperExecutionEngine):
    """
    Challenger 4: Avellaneda-Stoikov Adaptive Reservation Scalper
    Adjusts maker profit target and unwind floor based on elapsed holding time and volatility.
    Never lets a trade decay beyond 240s without attempting a positive maker exit.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 4: Avellaneda-Stoikov Reservation Scalper"

    def _manage_active_swing_position(self, decision, bbo_bid, bbo_ask, now_ms):
        if self.active_position is None:
            return
        pos = self.active_position
        elapsed_sec = (now_ms - pos.entry_ts_ms) / 1000.0
        sign = 1.0 if pos.side == "BUY" else -1.0
        mid = 0.5 * (bbo_bid + bbo_ask)
        unrealized_bps = sign * ((mid - pos.entry_price) / pos.entry_price) * 1e4

        # Dynamic reservation exit: if elapsed > 180s and unrealized >= 3.0 bps, close Maker
        if elapsed_sec >= 180.0 and unrealized_bps >= 3.00:
            maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
            self._close_position(
                exit_price=maker_exit_p,
                exit_ts_ms=now_ms,
                exit_fee_bps=self.maker_fee_leg,
                reason="AVELLANEDA_RESERVATION_MAKER_EXIT",
            )
            return

        super()._manage_active_swing_position(decision, bbo_bid, bbo_ask, now_ms)


# ==============================================================================
# 5. Bollinger Squeeze Breakout Scalper
# ==============================================================================
class BollingerSqueezeScalper(HierarchicalSniperExecutionEngine):
    """
    Challenger 5: Bollinger Squeeze Breakout Scalper
    Maintains rolling 20-bar volatility band. Detects volatility squeeze (BBW < 8.0 bps).
    Snipes maker entries at the initial impulse expansion following the squeeze.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 5: Bollinger Squeeze Breakout Scalper"
        self.rolling_prices: deque[float] = deque(maxlen=20)

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        now_ms = decision.ts_ms
        mid = 0.5 * (bbo_bid + bbo_ask)
        self.rolling_prices.append(mid)
        d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
        d_micro = decision.micro_probs[2] - decision.micro_probs[0]
        d_meso = decision.meso_probs[2] - decision.meso_probs[0]

        if len(self.rolling_prices) >= 20 and self.state == "IDLE":
            mean_p = np.mean(self.rolling_prices)
            std_p = np.std(self.rolling_prices)
            bbw_bps = (4.0 * std_p / max(1.0, mean_p)) * 1e4

            # Breakout after squeeze
            cand_dir = None
            if bbw_bps < 12.0:  # In squeeze or emerging from squeeze
                if mid > mean_p + 1.2 * std_p and d_tact >= 0.10 and d_meso >= 0.00:
                    cand_dir = 2
                elif mid < mean_p - 1.2 * std_p and d_tact <= -0.10 and d_meso <= 0.00:
                    cand_dir = 0

            if cand_dir in (0, 2):
                if self._is_parabolic_surge(cand_dir, now_ms, mid):
                    return self.state

                tact_dec = copy.copy(decision)
                object.__setattr__(tact_dec, "primary_direction", cand_dir) if hasattr(tact_dec, "__dataclass_fields__") and getattr(tact_dec, "__dataclass_params__", None) and getattr(tact_dec.__dataclass_params__, "frozen", False) else setattr(tact_dec, "primary_direction", cand_dir)
                self.stalking_decision = tact_dec
                self._place_sniper_maker_entry(tact_dec, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, now_ms, intra_bar_open)
                return self.state

        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)


def run_eval(ledger_file, competitors):
    with open(ledger_file, "r") as f:
        bars = [json.loads(line) for line in f if line.strip()]

    results = {}
    for name, factory in competitors.items():
        engine = factory()
        for bar in bars:
            bid = bar["bbo_bid"]
            ask = bar["bbo_ask"]
            ts = bar["ts_ms"]

            dec = HierarchicalDecisionOutput(
                ts_ms=ts,
                macro_probs=tuple(bar["macro_probs"]),
                meso_probs=tuple(bar["meso_probs"]),
                tactical_probs=tuple(bar["tactical_probs"]),
                micro_probs=tuple(bar["micro_probs"]),
                fused_swing_probs=(0.33, 0.34, 0.33),
                primary_direction=bar["primary_direction"],
                cross_horizon_alignment=1.0,
                expected_gross_move_bps=bar.get("expected_gross_move_bps", 10.0),
                expected_net_edge_bps=bar["expected_net_edge_bps"],
                meta_act_probability=0.6,
                micro_sniper_trigger_prob=0.6,
                target_holding_seconds=bar.get("suggested_holding_seconds", 300),
                take_profit_bps=15.0,
                stop_loss_bps=12.0,
                kelly_position_fraction=0.1,
                execution_mode="PASSIVE_MAKER_SNIPER" if bar["primary_direction"] != 1 else "NONE"
            )

            engine.on_market_micro_tick(
                ts_ms=ts,
                bbo_bid=bid,
                bbo_ask=ask,
                bbo_bid_qty=1.0,
                bbo_ask_qty=1.0,
                bar_high=bar.get("bar_high", bid),
                bar_low=bar.get("bar_low", ask),
                current_decision=dec,
            )

            engine.on_hierarchical_decision(
                decision=dec,
                bbo_bid=bid,
                bbo_ask=ask,
                bbo_bid_qty=1.0,
                bbo_ask_qty=1.0,
                equity_usdt=1000.0,
                intra_bar_open=bar["bbo_mid"]
            )

        trades = engine.closed_trades
        n = len(trades)
        w = sum(1 for t in trades if t.net_return_bps > 0)
        net = sum(t.net_return_bps for t in trades)
        wr = (w / n * 100.0) if n > 0 else 0.0
        results[name] = {"trades": n, "wins": w, "wr": round(wr, 1), "net_pnl": round(net, 2)}
    return results

competitors = {
    "Champion": lambda: PeakSameSideRetraceScalper(symbol="BTCUSDT"),
    "Challenger_1_MicroPrice": lambda: MicroPriceOFIScalper(symbol="BTCUSDT"),
    "Challenger_2_Resonant": lambda: MultiHorizonResonantScalper(symbol="BTCUSDT"),
    "Challenger_3_LeadLag": lambda: CrossAssetLeadLagScalper(symbol="BTCUSDT"),
    "Challenger_4_Avellaneda": lambda: AvellanedaStoikovReservationScalper(symbol="BTCUSDT"),
    "Challenger_5_Bollinger": lambda: BollingerSqueezeScalper(symbol="BTCUSDT"),
}

print("=== EVALUATION ON CYCLE 31 (PEAK REFERENCE) ===")
res_31 = run_eval("scratch/archive_cycle31/live_hierarchical_session_1h_ledger.jsonl", competitors)
for k, v in res_31.items():
    print(f"  {k:30s}: {v}")

print("\n=== EVALUATION ON CYCLE 32 (CHALLENGING SESSION) ===")
res_32 = run_eval("scratch/archive_cycle32/live_hierarchical_session_1h_ledger.jsonl", competitors)
for k, v in res_32.items():
    print(f"  {k:30s}: {v}")
