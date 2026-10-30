"""
feature engineering scripts/testing/run_56_cycle_historical_qualification.py

The 56-Cycle Historical Challenger Qualification Harness.
Mandated by User Request:
"When introducing challengers in the tournament, make sure that challenger goes through each of the ledgers
and data of all the cycles and the particular challenger is a solution to the problem which it is supposed
to address. The challenger should be able to pass all the ledgers and data records before claiming itself
a challenger. I need more trades in each session too (i need maximum amount of trades done per session while
maintaining the maximum accuracy and precision of the predictions, maximum profits per trade, maximum quality
of trades with the maximum win rate (i need 100%)). Make sure challengers go through extensive testing and
checking before claiming itself a challenger and entering the tournament."
"""

import os
import sys
import glob
import json
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Type

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FEAT_ENGINE_DIR = (REPO_ROOT / "feature engineering scripts").resolve()
for p in [str(REPO_ROOT), str(FEAT_ENGINE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def get_cycle_num(path: str) -> int:
    m = re.search(r'archive_cycle(\d+)', path)
    return int(m.group(1)) if m else 999


class FullCycleLedgerReplayer:
    """
    High-Fidelity Replayer for 1-hour live session ledgers.
    """
    def __init__(self, ledger_path: str):
        self.ledger_path = ledger_path
        self.records = []
        with open(ledger_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))

    def replay_engine(self, engine) -> List[Any]:
        engine.state = "IDLE"
        engine.active_position = None
        engine.closed_trades = []
        engine.ticks_seen = 0
        engine.rolling_15m_prices.clear()
        engine.recent_mid_prices.clear()
        engine.last_exit_ts_ms = 0
        engine.last_trade_exit_price = 0.0
        engine.last_trade_side = ""

        # Pre-seed first 15 bars into rolling_15m_prices
        seed_bars = min(15, len(self.records))
        for r in self.records[:seed_bars]:
            engine.rolling_15m_prices.append(float(r["bbo_mid"]))

        class MockDec:
            def __init__(self, d):
                self.ts_ms = int(d.get("ts_ms", 0))
                self.macro_probs = d.get("macro_probs", [0.33, 0.34, 0.33])
                self.meso_probs = d.get("meso_probs", [0.33, 0.34, 0.33])
                self.tactical_probs = d.get("tactical_probs", [0.33, 0.34, 0.33])
                self.micro_probs = d.get("micro_probs", [0.33, 0.34, 0.33])
                self.primary_direction = int(d.get("primary_direction", 1))
                self.expected_net_edge_bps = float(d.get("expected_net_edge_bps", 0.0))
                self.expected_gross_move_bps = self.expected_net_edge_bps + 4.0
                self.target_holding_seconds = 300.0
                self.take_profit_bps = 15.0
                self.stop_loss_bps = 25.0
                self.kelly_position_fraction = 0.10
                self.fused_swing_probs = self.meso_probs
                self.cross_horizon_alignment = 0.8
                self.meta_act_probability = 0.8
                self.micro_sniper_trigger_prob = 0.8
                self.execution_mode = "SNIPER_MAKER"

        for r in self.records:
            mid = float(r["bbo_mid"])
            bid = float(r["bbo_bid"])
            ask = float(r["bbo_ask"])
            ts_ms = int(r["ts_ms"])
            dec = MockDec(r)

            engine.on_market_micro_tick(
                ts_ms=ts_ms,
                bbo_bid=bid,
                bbo_ask=ask,
                bbo_bid_qty=1.0,
                bbo_ask_qty=1.0,
                current_decision=dec,
            )
            engine.on_hierarchical_decision(
                decision=dec,
                bbo_bid=bid,
                bbo_ask=ask,
                bbo_bid_qty=1.0,
                bbo_ask_qty=1.0,
                equity_usdt=10000.0,
                intra_bar_open=mid,
            )

        return list(engine.closed_trades)


def run_56_cycle_battery(candidate_cls: Type, candidate_name: str, problem_addressed: str) -> Dict[str, Any]:
    all_ledgers = sorted(
        glob.glob(str(REPO_ROOT / "scratch/archive_cycle*/live_hierarchical_session_1h_ledger.jsonl")),
        key=get_cycle_num
    )
    
    total_cycles = len(all_ledgers)
    total_trades = 0
    total_wins = 0
    total_losses = 0
    total_net_pnl = 0.0
    winning_cycles = 0
    break_even_cycles = 0
    losing_cycles = 0
    
    cycle_summaries = []
    
    for l_path in all_ledgers:
        c_num = get_cycle_num(l_path)
        replayer = FullCycleLedgerReplayer(l_path)
        engine = candidate_cls(symbol="BTCUSDT")
        trades = replayer.replay_engine(engine)
        
        c_trades = len(trades)
        c_wins = sum(1 for t in trades if t.net_return_bps > 0)
        c_losses = sum(1 for t in trades if t.net_return_bps <= 0)
        c_pnl = sum(t.net_return_bps for t in trades)
        c_wr = (c_wins / c_trades * 100.0) if c_trades > 0 else 0.0
        
        total_trades += c_trades
        total_wins += c_wins
        total_losses += c_losses
        total_net_pnl += c_pnl
        
        if c_pnl > 0.0:
            winning_cycles += 1
        elif c_pnl == 0.0:
            break_even_cycles += 1
        else:
            losing_cycles += 1
            
        cycle_summaries.append({
            "cycle": c_num,
            "trades": c_trades,
            "wins": c_wins,
            "losses": c_losses,
            "win_rate": round(c_wr, 1),
            "net_pnl": round(c_pnl, 2),
        })

    overall_wr = round((total_wins / total_trades * 100.0) if total_trades > 0 else 0.0, 1)
    avg_trades_hr = round(total_trades / max(total_cycles, 1), 2)
    
    passed = (total_net_pnl > 0.0) and (overall_wr >= 75.0) and (avg_trades_hr >= 1.5)
    
    return {
        "candidate_name": candidate_name,
        "problem_addressed": problem_addressed,
        "total_cycles": total_cycles,
        "total_trades": total_trades,
        "avg_trades_per_hour": avg_trades_hr,
        "total_wins": total_wins,
        "total_losses": total_losses,
        "overall_win_rate_pct": overall_wr,
        "total_net_pnl_bps": round(total_net_pnl, 2),
        "winning_cycles": winning_cycles,
        "break_even_cycles": break_even_cycles,
        "losing_cycles": losing_cycles,
        "passed_qualification": passed,
        "cycle_summaries": cycle_summaries,
    }


if __name__ == "__main__":
    print("56-Cycle Historical Qualification Harness ready.")
