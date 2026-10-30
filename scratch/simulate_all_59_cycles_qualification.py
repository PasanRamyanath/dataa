import os
import sys
import glob
import json
import copy
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Type

sys.path.insert(0, os.path.abspath('feature engineering scripts'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from execution.champion_challenger_framework import (
    Cycle35_ReversionPureScalper,
    Cycle62_MidpointVetoScalper,
    Cycle62_HighEdgeExpansionScalper,
    Cycle62_PureMicroMomentumScalper,
    Cycle62_SteppedRatchetScalper,
    Cycle62_PassiveUnwindMakerGuardScalper,
)


class AccurateRecordedDecision:
    def __init__(self, d: dict):
        self.ts_ms = int(d.get("ts_ms", 0))
        self.macro_probs = tuple(d.get("macro_probs", [0.33, 0.34, 0.33]))
        self.meso_probs = tuple(d.get("meso_probs", [0.33, 0.34, 0.33]))
        self.tactical_probs = tuple(d.get("tactical_probs", [0.33, 0.34, 0.33]))
        self.micro_probs = tuple(d.get("micro_probs", [0.33, 0.34, 0.33]))
        self.primary_direction = int(d.get("primary_direction", 1))
        self.expected_net_edge_bps = float(d.get("expected_net_edge_bps", 0.0))
        self.expected_gross_move_bps = self.expected_net_edge_bps + 4.0
        self.execution_mode = d.get("execution_mode", "NONE")
        self.target_holding_seconds = 300.0
        self.take_profit_bps = 15.0
        self.stop_loss_bps = 25.0
        self.kelly_position_fraction = 0.10
        self.fused_swing_probs = self.meso_probs
        self.cross_horizon_alignment = 0.8
        self.meta_act_probability = 0.8
        self.micro_sniper_trigger_prob = 0.8 if self.execution_mode == "SNIPER_MAKER" else 0.0


def replay_cycle_ledger(engine, ledger_path: str) -> List[dict]:
    """Replays a single cycle ledger with faithful state isolation and pre-seeding."""
    engine.state = "IDLE"
    engine.active_position = None
    engine.closed_trades = []
    engine.ticks_seen = 0
    engine.rolling_15m_prices.clear()
    engine.recent_mid_prices.clear()
    engine.last_exit_ts_ms = 0
    engine.last_trade_exit_price = 0.0
    engine.last_trade_side = ""
    engine.last_exit_was_win = False

    records = []
    with open(ledger_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return []

    # Pre-seed initial 15 bars to establish initial range
    seed_len = min(15, len(records))
    for r in records[:seed_len]:
        mid = float(r.get("bbo_mid", 0.0))
        if mid > 0:
            engine.rolling_15m_prices.append(mid)

    for r in records:
        mid = float(r.get("bbo_mid", 0.0))
        bid = float(r.get("bbo_bid", mid))
        ask = float(r.get("bbo_ask", mid))
        ts_ms = int(r.get("ts_ms", 0))
        dec = AccurateRecordedDecision(r)

        # Micro tick first
        engine.on_market_micro_tick(
            ts_ms=ts_ms,
            bbo_bid=bid,
            bbo_ask=ask,
            bbo_bid_qty=1.0,
            bbo_ask_qty=1.0,
            current_decision=dec,
        )

        # Hierarchical decision
        engine.on_hierarchical_decision(
            decision=dec,
            bbo_bid=bid,
            bbo_ask=ask,
            bbo_bid_qty=1.0,
            bbo_ask_qty=1.0,
            equity_usdt=10000.0,
            intra_bar_open=mid,
        )

    # Return summary dict of closed trades
    results = []
    for t in engine.closed_trades:
        results.append({
            "side": t.side,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "gross_return_bps": t.gross_return_bps,
            "net_return_bps": t.net_return_bps,
            "total_fee_bps": t.total_fee_bps,
            "exit_reason": t.exit_reason,
            "is_win": t.net_return_bps > 0.0,
        })
    return results


def run_full_simulation(engine_factory, engine_name: str, cycle_dirs: List[str]):
    total_trades = 0
    total_wins = 0
    total_losses = 0
    total_net_pnl = 0.0
    total_gross_pnl = 0.0
    total_fees = 0.0
    cycle_results = {}

    for c in cycle_dirs:
        ledger_path = os.path.join("scratch", c, "live_hierarchical_session_1h_ledger.jsonl")
        engine = engine_factory()
        trades = replay_cycle_ledger(engine, ledger_path)

        wins = sum(1 for t in trades if t["is_win"])
        losses = sum(1 for t in trades if not t["is_win"])
        net_pnl = sum(t["net_return_bps"] for t in trades)
        gross_pnl = sum(t["gross_return_bps"] for t in trades)
        fees = sum(t["total_fee_bps"] for t in trades)

        total_trades += len(trades)
        total_wins += wins
        total_losses += losses
        total_net_pnl += net_pnl
        total_gross_pnl += gross_pnl
        total_fees += fees

        cycle_results[c] = {
            "trades": len(trades),
            "wins": wins,
            "losses": losses,
            "win_rate": (wins / len(trades) * 100.0) if trades else 0.0,
            "net_pnl_bps": net_pnl,
        }

    wr = (total_wins / total_trades * 100.0) if total_trades > 0 else 0.0
    avg_trade = (total_net_pnl / total_trades) if total_trades > 0 else 0.0
    trades_per_hr = total_trades / len(cycle_dirs) if cycle_dirs else 0.0

    print(f"\n=======================================================")
    print(f"RESULTS FOR: {engine_name}")
    print(f"Cycles Evaluated: {len(cycle_dirs)}")
    print(f"Total Trades: {total_trades} | Wins: {total_wins} | Losses: {total_losses}")
    print(f"Aggregate Win Rate: {wr:.2f}%")
    print(f"Total Net PnL: {total_net_pnl:+.2f} bps")
    print(f"Total Gross PnL: {total_gross_pnl:+.2f} bps | Total Fees: {total_fees:.2f} bps")
    print(f"Avg Net Return per Trade: {avg_trade:+.2f} bps")
    print(f"Average Trades per 1h Session: {trades_per_hr:.2f}")
    print(f"=======================================================")
    return {
        "engine_name": engine_name,
        "total_trades": total_trades,
        "total_wins": total_wins,
        "total_losses": total_losses,
        "win_rate_pct": wr,
        "total_net_pnl_bps": total_net_pnl,
        "trades_per_hour": trades_per_hr,
        "cycle_results": cycle_results,
    }


if __name__ == "__main__":
    archives = [d for d in os.listdir("scratch") if d.startswith("archive_cycle") and os.path.isdir(os.path.join("scratch", d))]
    archives = sorted(archives, key=lambda x: int(x.replace("archive_cycle", "")))
    valid = [d for d in archives if os.path.exists(os.path.join("scratch", d, "live_hierarchical_session_1h_ledger.jsonl")) and os.path.getsize(os.path.join("scratch", d, "live_hierarchical_session_1h_ledger.jsonl")) > 0]
    print(f"Testing on all {len(valid)} verified historical cycles ({valid[0]} to {valid[-1]})...")

    # Benchmark Champion
    run_full_simulation(lambda: Cycle35_ReversionPureScalper(symbol="BTCUSDT"), "👑 Reigning Champion: Cycle 35 Baseline", valid)

    # Challenger 1: Midpoint Veto
    run_full_simulation(lambda: Cycle62_MidpointVetoScalper(symbol="BTCUSDT"), "⚔️ Challenger 1: Midpoint Veto Scalper", valid)
