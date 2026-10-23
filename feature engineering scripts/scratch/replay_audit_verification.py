"""
feature engineering scripts/scratch/replay_audit_verification.py
Replays the 720 bars from scratch/live_hierarchical_session_ledger.jsonl
through the fully remediated 7-point forensic audit architecture.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FEAT_ROOT = REPO_ROOT / "feature engineering scripts"
sys.path.insert(0, str(FEAT_ROOT))

from fusion.hierarchical_fusion_v3 import HierarchicalMultiHorizonFusionEngine
from execution.hierarchical_sniper_engine import HierarchicalSniperExecutionEngine
from streaming.run_live_hierarchical_session import ForwardAccuracyEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("replay_audit")


def run_replay(ledger_path: Path):
    log.info("Loading ledger from %s...", ledger_path)
    with open(ledger_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    log.info("Loaded %d bars for replay verification.", len(records))

    fusion = HierarchicalMultiHorizonFusionEngine()
    sniper = HierarchicalSniperExecutionEngine(symbol="BTCUSDT")
    evaluator = ForwardAccuracyEvaluator()

    flat_calls = 0
    active_calls = 0
    prev_decision = None

    for bar_idx, rec in enumerate(records, start=1):
        now_ms = int(rec.get("ts_ms", 0))
        bbo_mid = float(rec.get("bbo_mid", 0.0))
        bbo_bid = float(rec.get("bbo_bid", bbo_mid - 0.5))
        bbo_ask = float(rec.get("bbo_ask", bbo_mid + 0.5))
        bbo_bid_qty = 1.0
        bbo_ask_qty = 1.0

        p_macro = np.array(rec.get("macro_probs", [0.25, 0.50, 0.25]), dtype=np.float64)
        p_meso = np.array(rec.get("meso_probs", [0.25, 0.50, 0.25]), dtype=np.float64)
        p_tactical = np.array(rec.get("tactical_probs", [0.25, 0.50, 0.25]), dtype=np.float64)
        p_micro = np.array(rec.get("micro_probs", [0.25, 0.50, 0.25]), dtype=np.float64)

        # Dynamic ATRs
        atr_15m = 25.0
        atr_1h = 45.0
        atr_10s = 5.0

        # Run fused decision
        decision = fusion.evaluate_multi_horizon_tick(
            ts_ms=now_ms,
            p_macro=p_macro,
            p_meso=p_meso,
            p_tactical=p_tactical,
            p_micro=p_micro,
            atr_15m_bps=atr_15m,
            atr_1h_bps=atr_1h,
            atr_10s_bps=atr_10s,
            micro_safe_fill_prob=0.70,
        )

        if decision.primary_direction == 1:
            flat_calls += 1
        else:
            active_calls += 1

        # Advance Sniper FSM on market ticks & decision
        next_rec = records[bar_idx] if bar_idx < len(records) else rec
        next_ask = float(next_rec.get("bbo_ask", bbo_ask))
        next_bid = float(next_rec.get("bbo_bid", bbo_bid))
        h_price = max(bbo_ask, next_ask) + 0.2
        l_price = min(bbo_bid, next_bid) - 0.2

        sniper.on_market_micro_tick(
            ts_ms=now_ms,
            bbo_bid=bbo_bid,
            bbo_ask=bbo_ask,
            bbo_bid_qty=bbo_bid_qty,
            bbo_ask_qty=bbo_ask_qty,
            aggressive_sell_vol_at_bid=0.0,
            aggressive_buy_vol_at_ask=0.0,
            eth_1s_ret_bps=0.0,
            bar_high=h_price,
            bar_low=l_price,
            current_decision=decision,
        )
        sniper.on_hierarchical_decision(
            decision=decision,
            bbo_bid=bbo_bid,
            bbo_ask=bbo_ask,
            bbo_bid_qty=bbo_bid_qty,
            bbo_ask_qty=bbo_ask_qty,
            equity_usdt=10000.0,
            intra_bar_open=bbo_mid,
        )

        # Record tick in forward accuracy evaluator
        is_micro_auth = (decision.execution_mode in ("PASSIVE_MAKER_SNIPER", "URGENT_HYBRID"))
        evaluator.record_tick(
            bar_idx=bar_idx,
            ts_ms=now_ms,
            mid_price=bbo_mid,
            primary_dir=decision.primary_direction,
            macro_probs=decision.macro_probs,
            meso_probs=decision.meso_probs,
            tactical_probs=decision.tactical_probs,
            micro_probs=decision.micro_probs,
            micro_authorized=is_micro_auth,
            atr_15m_bps=atr_15m,
        )
        evaluator.update_evaluations(bar_idx, bbo_mid)
        prev_decision = decision

    summary = evaluator.get_summary()
    trades = sniper.closed_trades
    n_trades = len(trades)
    wins = sum(1 for t in trades if t.net_return_bps > 0)
    win_rate = (wins / n_trades * 100.0) if n_trades > 0 else 0.0
    net_pnl = sum(t.net_return_bps for t in trades)

    report = {
        "bars_replayed": len(records),
        "flat_calls": flat_calls,
        "active_calls": active_calls,
        "total_trades": n_trades,
        "winning_trades": wins,
        "win_rate_pct": round(win_rate, 2),
        "net_pnl_bps": round(net_pnl, 2),
        "forward_accuracy": summary,
        "closed_trades": [t.__dict__ for t in trades],
    }
    log.info("Replay Finished!")
    log.info("Trades: %d | Win Rate: %.1f%% | Net PnL: %+.2f bps", n_trades, win_rate, net_pnl)
    for tf, data in summary.items():
        if tf != "tiers":
            log.info("Horizon %s -> Acc: %.2f%% | Act Prec: %.2f%% (%d/%d) | Sign Acc: %.2f%% (%d/%d)",
                     tf, data["overall_accuracy_pct"], data["active_precision_pct"],
                     int(data["active_precision_pct"] * data["active_signals_count"] / 100.0), data["active_signals_count"],
                     data["sign_accuracy_pct"], int(data["sign_accuracy_pct"] * data["sign_samples"] / 100.0), data["sign_samples"])
    return report


if __name__ == "__main__":
    ledger = REPO_ROOT / "scratch" / "live_hierarchical_session_ledger.jsonl"
    res = run_replay(ledger)
    out_file = REPO_ROOT / "scratch" / "replay_audit_verification_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print("Verification report saved to:", out_file)
