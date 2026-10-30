import os
import sys
import copy
import json

sys.path.insert(0, os.path.abspath('feature engineering scripts'))
sys.path.insert(0, os.path.abspath('scratch'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from simulate_all_59_cycles_qualification import replay_cycle_ledger
from execution.champion_challenger_framework import (
    Cycle35_ReversionPureScalper,
    Cycle62_MidpointVetoScalper,
)

benchmark_suite = [
    "archive_cycle63",
    "archive_cycle62",
    "archive_cycle61",
    "archive_cycle60",
    "archive_cycle59",
    "archive_cycle58",
    "archive_cycle57",
    "archive_cycle35",
]

# Baseline engine (Current Champion)
class Baseline_Cycle35(Cycle35_ReversionPureScalper):
    pass

# Variant 1: Remove 0B2 (No premature Taker panic stop at -6.50 bps)
class Variant_No_0B2_TakerCut(Cycle35_ReversionPureScalper):
    """Disable 0B2 Taker stop completely. Allow trades to breathe or exit via Maker."""
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

        # [0B2 DUAL_HORIZON_INVALIDATION_TAKER REMOVED!]

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

# Variant 2: Replace 0B2 Taker with Passive Maker Unwind at wide -12.0 bps (only if elapsed >= 90s)
class Variant_PassiveMakerWideBuffer(Cycle35_ReversionPureScalper):
    """Replace 0B2 Taker stop with Passive Maker at -12.0 bps and elapsed >= 90s."""
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

        # 0B2_WideMaker: Wide Maker Invalidation at -12.0 bps and elapsed >= 90s
        if elapsed_sec >= 90.0 and unrealized_bps <= -12.0:
            maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
            self._close_position(exit_price=maker_exit_p, exit_ts_ms=now_ms, exit_fee_bps=self.maker_fee_leg, reason="PASSIVE_MAKER_WIDE_INVAL")
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

# Variant 3: Micro-Trailing Profit Shield + Wide Buffer (Peak exit engine)
class Variant_TrailingShield_NoUnderwaterCut(Cycle35_ReversionPureScalper):
    """Adds trailing profit protection when trade reaches +3.80 bps, but NEVER cuts underwater positions on noise."""
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

        # Dynamic Trailing Profit Shield
        if unrealized_bps >= 3.80:
            if not getattr(pos, 'trailing_armed', False):
                pos.trailing_armed = True
                pos.peak_excursion = unrealized_bps
            else:
                pos.peak_excursion = max(pos.peak_excursion, unrealized_bps)
            
            pullback = pos.peak_excursion - unrealized_bps
            # Lock profits: if pulled back by 2.20 bps from peak >= 6.0 bps, or drops below lock floor
            if pos.peak_excursion >= 6.50 and pullback >= 2.00:
                maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
                self._close_position(exit_price=maker_exit_p, exit_ts_ms=now_ms, exit_fee_bps=self.maker_fee_leg, reason="TRAILING_SHIELD_MAKER_TP")
                return

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


engines = {
    "Baseline (Cycle 60 Champ with 0B2 Taker Stop)": Baseline_Cycle35,
    "Variant 1 (Remove 0B2 Panic Taker Cut)": Variant_No_0B2_TakerCut,
    "Variant 2 (Replace 0B2 with Wide -12 bps Maker Exit >=90s)": Variant_PassiveMakerWideBuffer,
    "Variant 3 (Trailing Profit Shield + No Panic Taker Cut)": Variant_TrailingShield_NoUnderwaterCut,
}

print("="*95)
print("TESTING EXIT/INVALIDATION VARIABLE ACROSS 8 BENCHMARK CYCLES")
print("="*95)

for name, eng_cls in engines.items():
    total_trades = 0
    total_wins = 0
    total_net = 0.0
    total_fees = 0.0
    
    per_cycle_results = {}
    for cycle_dir in benchmark_suite:
        ledger_path = os.path.join("scratch", cycle_dir, "live_hierarchical_session_1h_ledger.jsonl")
        if not os.path.exists(ledger_path):
            continue
        eng = eng_cls(symbol="BTCUSDT")
        trades = replay_cycle_ledger(eng, ledger_path)
        c_trades = len(trades)
        c_wins = sum(1 for t in trades if t["is_win"])
        c_net = sum(t["net_return_bps"] for t in trades)
        c_fees = sum(t["total_fee_bps"] for t in trades)
        
        total_trades += c_trades
        total_wins += c_wins
        total_net += c_net
        total_fees += c_fees
        per_cycle_results[cycle_dir] = (c_trades, c_wins, c_net)
        
    wr = (total_wins / total_trades * 100.0) if total_trades > 0 else 0.0
    print(f"\n{name}")
    print(f"  TOTAL: Trades: {total_trades:2d} | Wins: {total_wins:2d} | WR: {wr:5.1f}% | Net: {total_net:+7.2f} bps | Fees: {total_fees:5.1f}b")
    for c_name, (c_tr, c_w, c_pnl) in per_cycle_results.items():
        print(f"    - {c_name}: {c_tr} trades, {c_w} wins, net {c_pnl:+6.2f} bps")


