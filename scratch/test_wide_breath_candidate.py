import os
import sys

sys.path.insert(0, os.path.abspath('feature engineering scripts'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from test_c63_candidates_59_cycles import Candidate1_MidpointRetestPassiveScalper, valid
from simulate_all_59_cycles_qualification import run_full_simulation


class Candidate_BreathingRoomScalper(Candidate1_MidpointRetestPassiveScalper):
    """
    Candidate with proper breathing room:
    1. Responsive Model Invalidation Maker Exit requires unrealized <= -6.50 bps (instead of premature -2.00 bps).
    2. Elapsed time for responsive exit requires >= 90s instead of 60s.
    3. Passive Maker Unwinds at BBO.
    4. Post-invalidation retest guard.
    5. Midpoint chop veto.
    """
    def __init__(self, symbol: str = "BTCUSDT"):
        super().__init__(symbol=symbol)
        self.strategy_profile = "Wide Breathing Room Passive Scalper"

    def _manage_active_swing_position(self, decision, bbo_bid, bbo_ask, now_ms):
        pos = self.active_position
        if pos is None:
            return

        elapsed_sec = (now_ms - pos.entry_ts_ms) / 1000.0
        sign = 1.0 if pos.side == "BUY" else -1.0
        current_mid = 0.5 * (bbo_bid + bbo_ask)
        unrealized_bps = sign * ((current_mid - pos.entry_price) / pos.entry_price) * 1e4

        if unrealized_bps > pos.peak_favorable_excursion_bps:
            pos.peak_favorable_excursion_bps = unrealized_bps
        if unrealized_bps < pos.max_adverse_excursion_bps:
            pos.max_adverse_excursion_bps = unrealized_bps

        d_meso = decision.meso_probs[2] - decision.meso_probs[0]
        d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
        d_micro = decision.micro_probs[2] - decision.micro_probs[0]
        opp_dir = 0 if pos.side == "BUY" else 2

        # 0. Full Take Profit Barrier Hit
        if unrealized_bps >= pos.take_profit_bps:
            maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
            self._close_position(
                exit_price=maker_exit_p,
                exit_ts_ms=now_ms,
                exit_fee_bps=self.maker_fee_leg,
                reason="RESTING_MAKER_TP",
            )
            return

        # 0A. Early Take-Profit Model Harvest
        if unrealized_bps >= self.model_harvest_min_bps:
            if (sign * d_micro) < -0.05 or decision.primary_direction == opp_dir:
                maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
                self._close_position(
                    exit_price=maker_exit_p,
                    exit_ts_ms=now_ms,
                    exit_fee_bps=self.maker_fee_leg,
                    reason="MODEL_HARVEST_MAKER_TP",
                )
                return

        # 0B1. Responsive Model Invalidation Maker Exit (WIDE BREATHING ROOM: requires unrealized <= -6.5 bps)
        if elapsed_sec >= 90.0 and unrealized_bps <= -6.50:
            meso_opposing = (sign * d_meso) < -0.12
            if meso_opposing or decision.primary_direction == opp_dir:
                maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
                self._close_position(
                    exit_price=maker_exit_p,
                    exit_ts_ms=now_ms,
                    exit_fee_bps=self.maker_fee_leg,
                    reason="RESPONSIVE_MODEL_MAKER_EXIT",
                )
                return

        # 0B2. Invalidation cut via Passive Maker Unwind at BBO (4.0 bps fee total)
        if unrealized_bps <= -6.50:
            tact_opposing = (sign * d_tact) < -0.08
            micro_opposing = (sign * d_micro) < -0.05
            meso_opposing = (sign * d_meso) < -0.10
            if (tact_opposing and micro_opposing) or meso_opposing or decision.primary_direction == opp_dir:
                maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
                self._close_position(
                    exit_price=maker_exit_p,
                    exit_ts_ms=now_ms,
                    exit_fee_bps=self.maker_fee_leg,
                    reason="PASSIVE_MAKER_UNWIND",
                )
                return

        # 1. Micro Trailing Maker TP
        if pos.peak_favorable_excursion_bps >= self.min_trailing_activation_bps:
            locked_floor_bps = max(self.tier1_min_lock_bps, pos.peak_favorable_excursion_bps * (1.0 - self.trailing_giveback_frac))
            if unrealized_bps <= locked_floor_bps:
                maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
                self._close_position(
                    exit_price=maker_exit_p,
                    exit_ts_ms=now_ms,
                    exit_fee_bps=self.maker_fee_leg,
                    reason="MICRO_TRAILING_MAKER_TP",
                )
                return

        # 2. Hard Stop Loss Protection
        if unrealized_bps <= -15.0:
            taker_exit_p = bbo_bid if pos.side == "BUY" else bbo_ask
            self._close_position(
                exit_price=taker_exit_p,
                exit_ts_ms=now_ms,
                exit_fee_bps=self.taker_fee_leg,
                reason="HARD_STOP_LOSS_TAKER",
            )
            return

        # 3. Passive Maker Time Unwind
        if elapsed_sec >= 300.0:
            maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
            self._close_position(
                exit_price=maker_exit_p,
                exit_ts_ms=now_ms,
                exit_fee_bps=self.maker_fee_leg,
                reason="PASSIVE_MAKER_TIME_UNWIND",
            )
            return


if __name__ == "__main__":
    print("Testing Candidate_BreathingRoomScalper across all 59 cycles...")
    run_full_simulation(lambda: Candidate_BreathingRoomScalper(), "Wide Breathing Room Passive Scalper", valid)
