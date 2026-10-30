import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, os.path.abspath('feature engineering scripts'))
sys.path.insert(0, os.path.abspath('scratch'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from simulate_all_59_cycles_qualification import replay_cycle_ledger, AccurateRecordedDecision
from execution.champion_challenger_framework import Cycle35_ReversionPureScalper

# Baseline
class Baseline_Cycle60(Cycle35_ReversionPureScalper):
    pass

# Candidate A: Remove 0B2 completely (No premature taker panic stop on sub-minute noise)
class Engine_No_0B2_Panic(Cycle35_ReversionPureScalper):
    def _manage_active_swing_position(self, decision, bbo_bid, bbo_ask, now_ms):
        if self.active_position is None:
            return
        pos = self.active_position
        elapsed_sec = (now_ms - pos.entry_ts_ms) / 1000.0
        hold_frac = elapsed_sec / max(1.0, float(pos.target_holding_seconds))

        sign = 1.0 if pos.side == "BUY" else -1.0
        mid = 0.5 * (bbo_bid + bbo_ask)
        unrealized_bps = sign * ((mid - pos.entry_price) / pos.entry_price) * 1e4
        d_macro = decision.macro_probs[2] - decision.macro_probs[0]
        d_meso = decision.meso_probs[2] - decision.meso_probs[0]
        d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
        d_micro = decision.micro_probs[2] - decision.micro_probs[0]
        opp_dir = 0 if pos.side == "BUY" else 2

        # 0. Full Take Profit Barrier Hit
        if unrealized_bps >= pos.take_profit_bps:
            maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
            self._close_position(exit_price=maker_exit_p, exit_ts_ms=now_ms, exit_fee_bps=self.maker_fee_leg, reason="RESTING_MAKER_TP")
            return

        # 0A. Active Neural Model Profit Harvest
        if unrealized_bps >= self.model_harvest_min_bps:
            if (sign * d_micro) < -0.05 or decision.primary_direction == opp_dir:
                maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
                self._close_position(exit_price=maker_exit_p, exit_ts_ms=now_ms, exit_fee_bps=self.maker_fee_leg, reason="MODEL_HARVEST_MAKER_TP")
                return

        # 0B1. Responsive Model Invalidation Maker Exit: (>= 60s, <= -2.00 bps)
        if elapsed_sec >= 60.0 and unrealized_bps <= -2.00:
            meso_opposing = (sign * d_meso) < -0.12
            if meso_opposing or decision.primary_direction == opp_dir:
                maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
                self._close_position(exit_price=maker_exit_p, exit_ts_ms=now_ms, exit_fee_bps=self.maker_fee_leg, reason="RESPONSIVE_MODEL_MAKER_EXIT")
                return

        # [0B2 Taker stop completely eliminated - noise filter]

        # 1. Macro/Meso violent reversal check
        if elapsed_sec >= 120.0 and unrealized_bps >= 4.30:
            if (sign * d_meso) < -0.22 and (sign * d_tact) < -0.15 and (sign * d_micro) < -0.10:
                maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
                self._close_position(exit_price=maker_exit_p, exit_ts_ms=now_ms, exit_fee_bps=self.maker_fee_leg, reason="MACRO_REVERSAL_MAKER_EXIT")
                return

        # 1B. Profitable Time-Decay & Opposite-Wave Maker Lock
        if elapsed_sec >= 300.0 and unrealized_bps >= self.passive_unwind_min_bps and ((sign * d_micro) <= 0.05 or decision.primary_direction == opp_dir):
            maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
            self._close_position(exit_price=maker_exit_p, exit_ts_ms=now_ms, exit_fee_bps=self.maker_fee_leg, reason="PASSIVE_MAKER_TIME_UNWIND")
            return

        # 2. Enter PASSIVE_MAKER_UNWIND at 85% of target holding horizon
        if hold_frac >= self.unwind_start_frac and self.state == "FILLED_SWING_HOLDING":
            self.state = "PASSIVE_MAKER_UNWIND"

        if self.state == "PASSIVE_MAKER_UNWIND":
            if (unrealized_bps >= self.passive_unwind_min_bps and ((sign * d_micro) <= 0.05 or hold_frac >= 0.95)) or (hold_frac >= 1.0 and unrealized_bps > 0.0):
                maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
                self._close_position(exit_price=maker_exit_p, exit_ts_ms=now_ms, exit_fee_bps=self.maker_fee_leg, reason="PASSIVE_MAKER_TIME_UNWIND")
                return

        # 3. Maximum holding duration guard:
        if elapsed_sec >= 600.0:
            maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
            self._close_position(exit_price=maker_exit_p, exit_ts_ms=now_ms, exit_fee_bps=self.maker_fee_leg, reason="PASSIVE_MAKER_TIME_UNWIND")


# Candidate B: True Thesis Invalidation: Only cut underwater if elapsed >= 90s and unrealized <= -12.0 bps via Maker BBO
class Engine_Thesis_Invalidation_Maker(Cycle35_ReversionPureScalper):
    def _manage_active_swing_position(self, decision, bbo_bid, bbo_ask, now_ms):
        if self.active_position is None:
            return
        pos = self.active_position
        elapsed_sec = (now_ms - pos.entry_ts_ms) / 1000.0
        hold_frac = elapsed_sec / max(1.0, float(pos.target_holding_seconds))

        sign = 1.0 if pos.side == "BUY" else -1.0
        mid = 0.5 * (bbo_bid + bbo_ask)
        unrealized_bps = sign * ((mid - pos.entry_price) / pos.entry_price) * 1e4
        d_macro = decision.macro_probs[2] - decision.macro_probs[0]
        d_meso = decision.meso_probs[2] - decision.meso_probs[0]
        d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
        d_micro = decision.micro_probs[2] - decision.micro_probs[0]
        opp_dir = 0 if pos.side == "BUY" else 2

        # 0. Full Take Profit Barrier Hit
        if unrealized_bps >= pos.take_profit_bps:
            maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
            self._close_position(exit_price=maker_exit_p, exit_ts_ms=now_ms, exit_fee_bps=self.maker_fee_leg, reason="RESTING_MAKER_TP")
            return

        # 0A. Active Neural Model Profit Harvest
        if unrealized_bps >= self.model_harvest_min_bps:
            if (sign * d_micro) < -0.05 or decision.primary_direction == opp_dir:
                maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
                self._close_position(exit_price=maker_exit_p, exit_ts_ms=now_ms, exit_fee_bps=self.maker_fee_leg, reason="MODEL_HARVEST_MAKER_TP")
                return

        # 0B1. Responsive Model Invalidation Maker Exit: (>= 60s, <= -2.00 bps)
        if elapsed_sec >= 60.0 and unrealized_bps <= -2.00:
            meso_opposing = (sign * d_meso) < -0.12
            if meso_opposing or decision.primary_direction == opp_dir:
                maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
                self._close_position(exit_price=maker_exit_p, exit_ts_ms=now_ms, exit_fee_bps=self.maker_fee_leg, reason="RESPONSIVE_MODEL_MAKER_EXIT")
                return

        # 0B2_Maker: True Invalidation (>= 90s elapsed AND <= -12.0 bps) via Maker order
        if elapsed_sec >= 90.0 and unrealized_bps <= -12.00:
            if (sign * d_meso) < -0.10 or decision.primary_direction == opp_dir:
                maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
                self._close_position(exit_price=maker_exit_p, exit_ts_ms=now_ms, exit_fee_bps=self.maker_fee_leg, reason="THESIS_INVALIDATION_MAKER")
                return

        # 1. Macro/Meso violent reversal check
        if elapsed_sec >= 120.0 and unrealized_bps >= 4.30:
            if (sign * d_meso) < -0.22 and (sign * d_tact) < -0.15 and (sign * d_micro) < -0.10:
                maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
                self._close_position(exit_price=maker_exit_p, exit_ts_ms=now_ms, exit_fee_bps=self.maker_fee_leg, reason="MACRO_REVERSAL_MAKER_EXIT")
                return

        # 1B. Profitable Time-Decay & Opposite-Wave Maker Lock
        if elapsed_sec >= 300.0 and unrealized_bps >= self.passive_unwind_min_bps and ((sign * d_micro) <= 0.05 or decision.primary_direction == opp_dir):
            maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
            self._close_position(exit_price=maker_exit_p, exit_ts_ms=now_ms, exit_fee_bps=self.maker_fee_leg, reason="PASSIVE_MAKER_TIME_UNWIND")
            return

        # 2. Enter PASSIVE_MAKER_UNWIND at 85% of target holding horizon
        if hold_frac >= self.unwind_start_frac and self.state == "FILLED_SWING_HOLDING":
            self.state = "PASSIVE_MAKER_UNWIND"

        if self.state == "PASSIVE_MAKER_UNWIND":
            if (unrealized_bps >= self.passive_unwind_min_bps and ((sign * d_micro) <= 0.05 or hold_frac >= 0.95)) or (hold_frac >= 1.0 and unrealized_bps > 0.0):
                maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
                self._close_position(exit_price=maker_exit_p, exit_ts_ms=now_ms, exit_fee_bps=self.maker_fee_leg, reason="PASSIVE_MAKER_TIME_UNWIND")
                return

        # 3. Maximum holding duration guard:
        if elapsed_sec >= 600.0:
            maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
            self._close_position(exit_price=maker_exit_p, exit_ts_ms=now_ms, exit_fee_bps=self.maker_fee_leg, reason="PASSIVE_MAKER_TIME_UNWIND")


if __name__ == "__main__":
    archives = [d for d in os.listdir("scratch") if d.startswith("archive_cycle") and os.path.isdir(os.path.join("scratch", d))]
    # Filter only numeric ones
    def cycle_num(d):
        try:
            return int(d.replace("archive_cycle", ""))
        except:
            return -1
    archives = sorted([d for d in archives if cycle_num(d) > 0], key=cycle_num)
    valid = [d for d in archives if os.path.exists(os.path.join("scratch", d, "live_hierarchical_session_1h_ledger.jsonl")) and os.path.getsize(os.path.join("scratch", d, "live_hierarchical_session_1h_ledger.jsonl")) > 1000]
    print(f"Loaded {len(valid)} verified historical cycles ({valid[0]} to {valid[-1]}).")

    engines = {
        "Baseline Cycle 60 (with 0B2 Panic Taker)": Baseline_Cycle60,
        "Engine No-0B2-Panic (Pure Maker Protection)": Engine_No_0B2_Panic,
        "Engine Thesis-Inval-Maker (>=90s & <= -12 bps)": Engine_Thesis_Invalidation_Maker,
    }

    for name, eng_cls in engines.items():
        total_trades = 0
        total_wins = 0
        total_net = 0.0
        total_gross = 0.0
        total_fees = 0.0
        prof_cycles = 0

        for c in valid:
            ledger_path = os.path.join("scratch", c, "live_hierarchical_session_1h_ledger.jsonl")
            eng = eng_cls(symbol="BTCUSDT")
            trades = replay_cycle_ledger(eng, ledger_path)
            c_net = sum(t["net_return_bps"] for t in trades)
            c_gross = sum(t["gross_return_bps"] for t in trades)
            c_fees = sum(t["total_fee_bps"] for t in trades)
            c_wins = sum(1 for t in trades if t["is_win"])

            total_trades += len(trades)
            total_wins += c_wins
            total_net += c_net
            total_gross += c_gross
            total_fees += c_fees
            if c_net > 0:
                prof_cycles += 1

        wr = (total_wins / total_trades * 100.0) if total_trades > 0 else 0.0
        print(f"\n=======================================================")
        print(f"ENGINE: {name}")
        print(f"Total Cycles: {len(valid)} | Profitable Cycles: {prof_cycles}/{len(valid)} ({prof_cycles/len(valid)*100.0:.1f}%)")
        print(f"Trades: {total_trades} | Wins: {total_wins} | Aggregate Win Rate: {wr:.2f}%")
        print(f"Total Gross PnL: {total_gross:+.2f} bps | Total Fees: {total_fees:.2f} bps")
        print(f"TOTAL NET PnL: {total_net:+.2f} bps")
        print(f"Avg Net PnL/Trade: {(total_net/total_trades):+.2f} bps | Trades/Hour: {(total_trades/len(valid)):.2f}")
        print(f"=======================================================")
