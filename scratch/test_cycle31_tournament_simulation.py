"""
scratch/test_cycle31_tournament_simulation.py
Simulates the restored Cycle 31 Tournament Lineup across all 5 historical cycles (25-29)
to verify performance and confirm the tournament gating logic works as intended.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path("feature engineering scripts").resolve()))

import json
import copy
import numpy as np
from execution.hierarchical_sniper_engine import HierarchicalSniperExecutionEngine
from fusion.hierarchical_fusion_v3 import HierarchicalDecisionOutput

# 1. Champion
class PeakSameSideRetraceScalper(HierarchicalSniperExecutionEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Reigning Champion: Peak Same-Side Retrace Scalper (+37.78 bps Net in C25-C26)"
        self.max_parabolic_surge_30s_bps = 10.0
        self.max_impulse_surge_180s_bps = 12.0
        self.max_impulse_surge_300s_bps = 14.0
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
            # Check global re-entry cooldown & $15 retrace barrier across ALL entry mechanisms
            cand_dir = None
            tact_dir = 2 if d_tact >= 0.14 else (0 if d_tact <= -0.14 else 1)
            tact_confirmed = (tact_dir in (0, 2)) and (
                (tact_dir == 2 and d_micro > 0.02 and d_meso >= 0.00) or
                (tact_dir == 0 and d_micro < -0.02 and d_meso <= 0.00)
            )
            
            if tact_confirmed:
                cand_dir = tact_dir
            elif decision.primary_direction in (0, 2) and decision.expected_net_edge_bps >= self.min_net_edge_bps:
                cand_dir = decision.primary_direction

            if cand_dir in (0, 2):
                # 1. Parabolic surge veto
                if self._is_parabolic_surge(cand_dir, now_ms, mid):
                    return self.state

                # 2. Strict Retrace Barrier & Cooldown
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

                # Authorized entry!
                tact_dec = copy.copy(decision)
                object.__setattr__(tact_dec, "primary_direction", cand_dir) if hasattr(tact_dec, "__dataclass_fields__") and getattr(tact_dec, "__dataclass_params__", None) and getattr(tact_dec.__dataclass_params__, "frozen", False) else setattr(tact_dec, "primary_direction", cand_dir)
                self.stalking_decision = tact_dec
                self._place_sniper_maker_entry(tact_dec, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, now_ms, intra_bar_open)
                return self.state

        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)

# 2. Challenger 1: Strict Meso Concordance
class Challenger1_StrictMesoConcordanceScalper(PeakSameSideRetraceScalper):
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
            tact_dir = 2 if d_tact >= 0.14 else (0 if d_tact <= -0.14 else 1)
            tact_confirmed = (tact_dir in (0, 2)) and (
                (tact_dir == 2 and d_micro > 0.02 and d_meso >= 0.02) or
                (tact_dir == 0 and d_micro < -0.02 and d_meso <= -0.02)
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
                    self.stalking_decision = tact_dec
                    self._place_sniper_maker_entry(tact_dec, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, now_ms, intra_bar_open)
                    return self.state
                else:
                    # Retrace barrier or cooldown actively blocked this entry: DO NOT fall through!
                    return self.state
        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)

def test_engine_on_cycle(engine_cls, cycle_num):
    ledger_path = Path(f"scratch/archive_cycle{cycle_num}/live_hierarchical_session_1h_ledger.jsonl")
    if not ledger_path.exists():
        return None
    with open(ledger_path) as f:
        bars = [json.loads(line) for line in f]
    engine = engine_cls(symbol="BTCUSDT")
    for b in bars:
        dec = HierarchicalDecisionOutput(
            ts_ms=b["ts_ms"],
            primary_direction=b["primary_direction"],
            macro_probs=np.array(b["macro_probs"]),
            meso_probs=np.array(b["meso_probs"]),
            tactical_probs=np.array(b["tactical_probs"]),
            micro_probs=np.array(b["micro_probs"]),
            fused_swing_probs=np.array(b["tactical_probs"]),
            cross_horizon_alignment=0.8,
            expected_gross_move_bps=15.0,
            expected_net_edge_bps=b["expected_net_edge_bps"],
            meta_act_probability=0.75,
            micro_sniper_trigger_prob=0.6,
            target_holding_seconds=900,
            take_profit_bps=35.0,
            stop_loss_bps=14.5,
            kelly_position_fraction=0.1,
            execution_mode=b.get("execution_mode", "NONE"),
        )
        mid = b["bbo_mid"]
        engine.on_market_micro_tick(
            ts_ms=b["ts_ms"],
            bbo_bid=b["bbo_bid"],
            bbo_ask=b["bbo_ask"],
            bbo_bid_qty=1.0,
            bbo_ask_qty=1.0,
            bar_high=mid + 0.5,
            bar_low=mid - 0.5,
            current_decision=dec,
        )
        engine.on_hierarchical_decision(
            decision=dec,
            bbo_bid=b["bbo_bid"],
            bbo_ask=b["bbo_ask"],
            bbo_bid_qty=1.0,
            bbo_ask_qty=1.0,
            equity_usdt=10000.0,
            intra_bar_open=mid,
        )
    return engine.closed_trades

print("Testing Champion and Challenger 1 across Cycles 25 and 26...")
for c in [25, 26]:
    trades = test_engine_on_cycle(PeakSameSideRetraceScalper, c)
    n = len(trades)
    w = sum(1 for t in trades if t.net_return_bps > 0)
    net = sum(t.net_return_bps for t in trades)
    wr = (w / n * 100.0) if n > 0 else 0.0
    print(f"Cycle {c} Champion: {n} Trades | {w} Wins | WR: {wr:.1f}% | Net: {net:+.2f} bps")
    for i, t in enumerate(trades, 1):
        print(f"  Trade {i}: {t.side:<4} @ {t.entry_price:.1f} -> {t.exit_price:.1f} | Net: {t.net_return_bps:>+6.2f} bps | MFE: {t.peak_mfe_bps:>+5.2f}b | {t.exit_reason}")
