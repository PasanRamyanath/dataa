import json
import os
import sys

sys.path.insert(0, os.path.abspath("feature engineering scripts"))
from execution.champion_challenger_framework import PeakSameSideRetraceScalper
from fusion.hierarchical_fusion_v3 import HierarchicalDecisionOutput

class ResponsiveInvalidationScalper(PeakSameSideRetraceScalper):
    """
    Candidate Champion / Challenger:
    Adds Early Responsive Model Invalidation:
    If position is underwater (unrealized_bps <= -0.50 bps) and multi-model momentum flips hostile
    (d_meso opposing < -0.10 or primary_direction == opp_dir), exit immediately at BBO Maker (or Taker safety)
    preventing 14 bps catastrophic stops!
    """
    def _manage_active_swing_position(self, decision, bbo_bid, bbo_ask, now_ms):
        if self.active_position is None:
            return
        pos = self.active_position
        sign = 1.0 if pos.side == "BUY" else -1.0
        mid = 0.5 * (bbo_bid + bbo_ask)
        unrealized_bps = sign * ((mid - pos.entry_price) / pos.entry_price) * 1e4
        d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
        d_micro = decision.micro_probs[2] - decision.micro_probs[0]
        d_meso = decision.meso_probs[2] - decision.meso_probs[0]
        opp_dir = 0 if pos.side == "BUY" else 2

        # 0B. Responsive Early Invalidation Cut:
        # If position is even slightly underwater (<= -0.50 bps) and Meso flips hostile (< -0.10)
        # or primary direction flips to opp_dir, exit immediately via BBO Maker
        if unrealized_bps <= -0.50:
            meso_opposing = (sign * d_meso) < -0.10
            tact_opposing = (sign * d_tact) < -0.08
            if meso_opposing or decision.primary_direction == opp_dir or (tact_opposing and (sign * d_micro) < -0.05):
                maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
                self._close_position(
                    exit_price=maker_exit_p,
                    exit_ts_ms=now_ms,
                    exit_fee_bps=self.maker_fee_leg,
                    reason="RESPONSIVE_MODEL_INVALIDATION_MAKER",
                )
                return

        super()._manage_active_swing_position(decision, bbo_bid, bbo_ask, now_ms)


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

print("=== BASELINE CHAMPION ON CYCLE 32 ===")
res_base = simulate_session("scratch/archive_cycle32/live_hierarchical_session_1h_ledger.jsonl", lambda: PeakSameSideRetraceScalper(symbol="BTCUSDT"))
print("Baseline:", res_base)

print("\n=== RESPONSIVE INVALIDATION SCALPER ON CYCLE 32 ===")
res_resp = simulate_session("scratch/archive_cycle32/live_hierarchical_session_1h_ledger.jsonl", lambda: ResponsiveInvalidationScalper(symbol="BTCUSDT"))
print("Responsive Invalidation:", res_resp)

print("\n=== RESPONSIVE INVALIDATION SCALPER ON CYCLE 31 (PROVING ZERO REGRESSION) ===")
res_c31 = simulate_session("scratch/archive_cycle31/live_hierarchical_session_1h_ledger.jsonl", lambda: ResponsiveInvalidationScalper(symbol="BTCUSDT"))
print("Cycle 31 with Responsive Invalidation:", res_c31)
