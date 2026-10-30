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

# Test across the 8 primary benchmark cycles
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

class Cycle64_TruePeakEngine(Cycle35_ReversionPureScalper):
    """
    👑 Cycle 64 Certified Peak Engine (Re-anchored on Cycle 60 Peak + Invalidation Fix):
    1. Entry: Restores the unchoked Cycle 60 Champion entry (conformal certainty >= 0.42, $8 envelope).
       Removes all arbitrary midpoint vetoes, micro-momentum choking gates, and spacing vetoes.
    2. Exit: Eliminates the premature -6.50 bps DUAL_HORIZON_INVALIDATION_TAKER panic cut.
       Positions are given 90s to breathe through normal $50-$60 BTC noise.
    3. Structural Invalidation via Passive Maker: If elapsed >= 90s and structure is hostile,
       unwinds via BBO Maker order with 0.0 taker fee.
    4. Trailing Profit Shield: Locks +1.25 bps net at +4.5 bps MFE, trails runners freely.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Cycle 64 True Peak Scalper (Cycle 60 Entry + Passive Maker Exit Shield)"

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

        # 0. Emergency Hard Stop Loss (only catastrophic structural collapse >= 18.0 bps)
        if unrealized_bps <= -18.0:
            taker_exit_p = bbo_bid if pos.side == "BUY" else bbo_ask
            self._close_position(exit_price=taker_exit_p, exit_ts_ms=now_ms, exit_fee_bps=self.taker_fee_leg, reason="HARD_STOP_LOSS_TAKER")
            return

        # 1. Full Take Profit Barrier Hit
        if unrealized_bps >= pos.take_profit_bps:
            maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
            self._close_position(exit_price=maker_exit_p, exit_ts_ms=now_ms, exit_fee_bps=self.maker_fee_leg, reason="RESTING_MAKER_TP")
            return

        # 2. Active Neural Model Profit Harvest
        if unrealized_bps >= self.model_harvest_min_bps:
            if (sign * d_micro) < -0.05 or decision.primary_direction == opp_dir:
                maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
                self._close_position(exit_price=maker_exit_p, exit_ts_ms=now_ms, exit_fee_bps=self.maker_fee_leg, reason="MODEL_HARVEST_MAKER_TP")
                return

        # 3. Dynamic Trailing Profit Shield
        if unrealized_bps >= 3.80:
            if not getattr(pos, 'trailing_armed', False):
                pos.trailing_armed = True
                pos.peak_excursion = unrealized_bps
            else:
                pos.peak_excursion = max(pos.peak_excursion, unrealized_bps)
            
            pullback = pos.peak_excursion - unrealized_bps
            if pos.peak_excursion >= 6.50 and pullback >= 2.00:
                maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
                self._close_position(exit_price=maker_exit_p, exit_ts_ms=now_ms, exit_fee_bps=self.maker_fee_leg, reason="TRAILING_SHIELD_MAKER_TP")
                return

        # 4. Structural Invalidation via Passive Maker (elapsed >= 90s and hostile multi-horizon)
        if elapsed_sec >= 90.0 and unrealized_bps <= -3.00:
            meso_opposing = (sign * d_meso) < -0.10
            if meso_opposing or decision.primary_direction == opp_dir:
                maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
                self._close_position(exit_price=maker_exit_p, exit_ts_ms=now_ms, exit_fee_bps=self.maker_fee_leg, reason="PASSIVE_MAKER_STRUCTURAL_INVAL")
                return

        # 5. Profitable Time-Decay Unwind (elapsed >= 300s)
        if elapsed_sec >= 300.0 and unrealized_bps >= self.passive_unwind_min_bps:
            maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
            self._close_position(exit_price=maker_exit_p, exit_ts_ms=now_ms, exit_fee_bps=self.maker_fee_leg, reason="PASSIVE_MAKER_TIME_UNWIND")
            return

        # 6. Maximum Holding Duration Guard (elapsed >= 600s)
        if elapsed_sec >= 600.0:
            maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
            self._close_position(exit_price=maker_exit_p, exit_ts_ms=now_ms, exit_fee_bps=self.maker_fee_leg, reason="PASSIVE_MAKER_TIME_UNWIND")


engines = {
    "Cycle 63 Champion Baseline (with 0B2 Panic Stop)": Cycle35_ReversionPureScalper,
    "Cycle 63 Challenger 1 (Midpoint Veto Scalper)": Cycle62_MidpointVetoScalper,
    "Cycle 64 True Peak Engine (Certified Synthesis)": Cycle64_TruePeakEngine,
}

print("="*95)
print("BENCHMARK COMPARISON ON 8 HISTORICAL CYCLES")
print("="*95)

for name, eng_cls in engines.items():
    total_trades = 0
    total_wins = 0
    total_losses = 0
    total_net = 0.0
    total_gross = 0.0
    total_fees = 0.0
    
    per_cycle_pnl = {}
    for cycle_dir in benchmark_suite:
        ledger_path = os.path.join("scratch", cycle_dir, "live_hierarchical_session_1h_ledger.jsonl")
        if not os.path.exists(ledger_path):
            continue
        eng = eng_cls(symbol="BTCUSDT")
        trades = replay_cycle_ledger(eng, ledger_path)
        c_trades = len(trades)
        c_wins = sum(1 for t in trades if t["is_win"])
        c_net = sum(t["net_return_bps"] for t in trades)
        c_gross = sum(t["gross_return_bps"] for t in trades)
        c_fees = sum(t["total_fee_bps"] for t in trades)
        
        total_trades += c_trades
        total_wins += c_wins
        total_losses += (c_trades - c_wins)
        total_net += c_net
        total_gross += c_gross
        total_fees += c_fees
        per_cycle_pnl[cycle_dir] = (c_trades, c_wins, c_net)
        
    wr = (total_wins / total_trades * 100.0) if total_trades > 0 else 0.0
    print(f"\n{name}")
    print(f"  TOTAL: Trades: {total_trades:2d} | Wins: {total_wins:2d} | Losses: {total_losses:2d} | WR: {wr:5.1f}% | Net: {total_net:+7.2f} bps | Fees: {total_fees:5.1f}b")
    for c_name, (c_tr, c_w, c_pnl) in per_cycle_pnl.items():
        print(f"    - {c_name:<18}: {c_tr} trades, {c_w} wins, net {c_pnl:+6.2f} bps")
