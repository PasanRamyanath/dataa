"""
scratch/test_pullback_maker.py
Tests Pullback Discount Maker Scalper on Cycles 25, 26, 31, 33, 34
"""
import sys
sys.path.insert(0, "feature engineering scripts")
import json
from pathlib import Path
from execution.champion_challenger_framework import PeakSameSideRetraceScalper
from fusion.hierarchical_fusion_v3 import HierarchicalDecisionOutput

class PullbackDiscountMakerScalper(PeakSameSideRetraceScalper):
    def __init__(self, discount_dollars=1.50, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount_dollars = discount_dollars
        self.strategy_profile = f"Pullback Discount Maker Scalper (${discount_dollars:.2f} Discount)"

    def _place_sniper_maker_entry(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, now_ms, intra_bar_open=0.0):
        side = "BUY" if decision.primary_direction == 2 else "SELL"
        # Step limit price back by discount_dollars for favorable fill
        if side == "BUY":
            limit_price = round(bbo_bid - self.discount_dollars, 2)
        else:
            limit_price = round(bbo_ask + self.discount_dollars, 2)

        supporting_qty = bbo_bid_qty if side == "BUY" else bbo_ask_qty
        notional = max(100.0, equity_usdt * max(0.10, decision.kelly_position_fraction))
        qty = round(notional / limit_price, 3)

        self.resting_entry_side = side
        self.resting_entry_price = limit_price
        self.resting_entry_qty = qty
        self.resting_entry_ts_ms = now_ms
        from execution.hierarchical_sniper_engine import QueuePriorityTracker, AdverseSelectionGuard
        self.queue_tracker = QueuePriorityTracker(initial_queue_ahead=0.0)
        self.adverse_guard = AdverseSelectionGuard(initial_supporting_depth=supporting_qty)
        self.state = "RESTING_ENTRY_QUEUE"

def run_test():
    for c in [25, 26, 31, 33, 34]:
        ledger_path = Path(f"scratch/archive_cycle{c}/live_hierarchical_session_1h_ledger.jsonl")
        if not ledger_path.exists():
            continue
        bars = []
        with open(ledger_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    bars.append(json.loads(line))

        engine = PullbackDiscountMakerScalper(discount_dollars=2.0, symbol="BTCUSDT")
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
            engine.on_market_micro_tick(
                ts_ms=b["ts_ms"],
                bbo_bid=b["bbo_bid"],
                bbo_ask=b["bbo_ask"],
                bbo_bid_qty=b.get("bbo_bid_qty", 1.5),
                bbo_ask_qty=b.get("bbo_ask_qty", 1.5),
                bar_high=b.get("bar_high", b["bbo_ask"]),
                bar_low=b.get("bar_low", b["bbo_bid"]),
                current_decision=dec,
            )
            engine.on_hierarchical_decision(
                decision=dec,
                bbo_bid=b["bbo_bid"],
                bbo_ask=b["bbo_ask"],
                bbo_bid_qty=b.get("bbo_bid_qty", 1.5),
                bbo_ask_qty=b.get("bbo_ask_qty", 1.5),
                equity_usdt=10000.0,
                intra_bar_open=b.get("intra_bar_open", b["bbo_bid"]),
            )

        tr = engine.closed_trades
        n_tr = len(tr)
        wins = sum(1 for t in tr if t.net_return_bps > 0)
        wr = (wins / n_tr * 100) if n_tr > 0 else 0.0
        tot_net = sum(t.net_return_bps for t in tr)
        print(f"Cycle {c:2d} | Trades: {n_tr} | Wins: {wins} | Win Rate: {wr:5.1f}% | Net PnL: {tot_net:+6.2f} bps")
        for i, t in enumerate(tr, 1):
            print(f"   T{i}: {t.side} @ {t.entry_price:.2f} -> {t.exit_price:.2f} | Net: {t.net_return_bps:+6.2f} bps | MFE: {t.peak_mfe_bps:+5.2f}b | Hold: {t.holding_duration_sec:.0f}s | Reason: {t.exit_reason}")

if __name__ == "__main__":
    run_test()
