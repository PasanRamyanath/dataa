import json
import os
import sys

sys.path.insert(0, os.path.abspath("feature engineering scripts"))
from execution.champion_challenger_framework import PeakSameSideRetraceScalper
from fusion.hierarchical_fusion_v3 import HierarchicalDecisionOutput
import copy

class QuadHorizonPeakScalper(PeakSameSideRetraceScalper):
    """
    Candidate Peak Engine: Quad-Horizon Concordant Scalper
    Requires Quad-Horizon agreement to enter:
    - BUY: d_tact >= 0.10, d_micro >= 0.015, d_meso >= 0.02, d_macro >= 0.00
    - SELL: d_tact <= -0.10, d_micro <= -0.015, d_meso <= -0.02, d_macro <= 0.00
    Eliminates false entries when Tactical or Micro is flat or opposing!
    """
    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        now_ms = decision.ts_ms
        mid = 0.5 * (bbo_bid + bbo_ask)
        d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
        d_micro = decision.micro_probs[2] - decision.micro_probs[0]
        d_meso = decision.meso_probs[2] - decision.meso_probs[0]
        d_macro = decision.macro_probs[2] - decision.macro_probs[0]

        if self.state == "IDLE" and self.ticks_seen >= 10:
            cand_dir = None
            # Quad-Horizon Concordance Gating
            if d_tact >= 0.10 and d_micro >= 0.015 and d_meso >= 0.02 and d_macro >= 0.00:
                cand_dir = 2
            elif d_tact <= -0.10 and d_micro <= -0.015 and d_meso <= -0.02 and d_macro <= 0.00:
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
            else:
                return self.state

        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)


def simulate_session(ledger_file, engine_factory):
    with open(ledger_file, "r") as f:
        bars = [json.loads(line) for line in f if line.strip()]

    engine = engine_factory()
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
    return {
        "trades": n,
        "wins": w,
        "win_rate": wr,
        "net_pnl": net,
        "trade_details": [
            {
                "side": t.side,
                "net": round(t.net_return_bps, 2),
                "mfe": round(t.peak_mfe_bps, 2),
                "mae": round(t.max_mae_bps, 2),
                "dur": t.holding_duration_sec,
                "exit": t.exit_reason
            }
            for t in trades
        ]
    }

print("=== CYCLE 32 WITH QUAD-HORIZON SCALPER ===")
res_c32 = simulate_session("scratch/archive_cycle32/live_hierarchical_session_1h_ledger.jsonl", lambda: QuadHorizonPeakScalper(symbol="BTCUSDT"))
print("Result Cycle 32:", res_c32)

print("\n=== CYCLE 31 WITH QUAD-HORIZON SCALPER ===")
res_c31 = simulate_session("scratch/archive_cycle31/live_hierarchical_session_1h_ledger.jsonl", lambda: QuadHorizonPeakScalper(symbol="BTCUSDT"))
print("Result Cycle 31:", res_c31)

print("\n=== CYCLE 26 WITH QUAD-HORIZON SCALPER ===")
res_c26 = simulate_session("scratch/archive_cycle26/live_hierarchical_session_1h_ledger.jsonl", lambda: QuadHorizonPeakScalper(symbol="BTCUSDT"))
print("Result Cycle 26:", res_c26)

print("\n=== CYCLE 25 WITH QUAD-HORIZON SCALPER ===")
res_c25 = simulate_session("scratch/archive_cycle25/live_hierarchical_session_1h_ledger.jsonl", lambda: QuadHorizonPeakScalper(symbol="BTCUSDT"))
print("Result Cycle 25:", res_c25)
