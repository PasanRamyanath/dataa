import os
import sys

sys.path.insert(0, os.path.abspath('feature engineering scripts'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from test_c64_candidates_benchmark import (
    Cycle64_CompositeRetestPassiveScalper,
    benchmark_suite,
    replay_cycle_ledger,
)


class Cycle64_CalibratedBufferScalper(Cycle64_CompositeRetestPassiveScalper):
    """
    ⚔️ Challenger 2: Calibrated Invalidation Buffer Scalper
    Widens invalidation threshold from -6.50 bps to -8.50 bps so normal tick spread doesn't trigger false cuts.
    """
    def __init__(self, symbol: str = "BTCUSDT"):
        super().__init__(symbol=symbol)
        self.strategy_profile = "Challenger 2: Calibrated Buffer Scalper (-8.5 bps Stop Buffer)"

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

        # Full Take Profit
        if unrealized_bps >= pos.take_profit_bps:
            maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
            self._close_position(maker_exit_p, now_ms, self.maker_fee_leg, "RESTING_MAKER_TP")
            return

        # Model Harvest TP
        if unrealized_bps >= self.model_harvest_min_bps:
            if (sign * d_micro) < -0.05 or decision.primary_direction == opp_dir:
                maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
                self._close_position(maker_exit_p, now_ms, self.maker_fee_leg, "MODEL_HARVEST_MAKER_TP")
                return

        # Calibrated Invalidation: requires unrealized <= -8.50 bps (instead of -6.50 bps)
        if unrealized_bps <= -8.50:
            tact_opposing = (sign * d_tact) < -0.08
            micro_opposing = (sign * d_micro) < -0.05
            meso_opposing = (sign * d_meso) < -0.10
            if (tact_opposing and micro_opposing) or meso_opposing or decision.primary_direction == opp_dir:
                maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
                self._close_position(maker_exit_p, now_ms, self.maker_fee_leg, "PASSIVE_MAKER_UNWIND")
                return

        # Trailing TP
        if pos.peak_favorable_excursion_bps >= self.min_trailing_activation_bps:
            locked_floor = max(self.tier1_min_lock_bps, pos.peak_favorable_excursion_bps * (1.0 - self.trailing_giveback_frac))
            if unrealized_bps <= locked_floor:
                maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
                self._close_position(maker_exit_p, now_ms, self.maker_fee_leg, "MICRO_TRAILING_MAKER_TP")
                return

        # Hard SL
        if unrealized_bps <= -15.0:
            taker_exit_p = bbo_bid if pos.side == "BUY" else bbo_ask
            self._close_position(taker_exit_p, now_ms, self.taker_fee_leg, "HARD_STOP_LOSS_TAKER")
            return

        # Passive Time Unwind
        if elapsed_sec >= 300.0:
            maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
            self._close_position(maker_exit_p, now_ms, self.maker_fee_leg, "PASSIVE_MAKER_TIME_UNWIND")
            return


if __name__ == "__main__":
    tot_trades = 0
    tot_wins = 0
    tot_losses = 0
    tot_net_pnl = 0.0
    tot_fees = 0.0

    for c in benchmark_suite:
        ledger_path = os.path.join("scratch", c, "live_hierarchical_session_1h_ledger.jsonl")
        eng = Cycle64_CalibratedBufferScalper()
        trades = replay_cycle_ledger(eng, ledger_path)
        w = sum(1 for t in trades if t["is_win"])
        l = sum(1 for t in trades if not t["is_win"])
        net = sum(t["net_return_bps"] for t in trades)
        fees = sum(t["total_fee_bps"] for t in trades)
        tot_trades += len(trades)
        tot_wins += w
        tot_losses += l
        tot_net_pnl += net
        tot_fees += fees

    wr = (tot_wins / tot_trades * 100.0) if tot_trades > 0 else 0.0
    print(f"Calibrated Buffer Scalper | Trades: {tot_trades:2d} | Wins: {tot_wins:2d} | Losses: {tot_losses:2d} | WR: {wr:5.1f}% | Net: {tot_net_pnl:+6.2f} bps | Fees: {tot_fees:5.1f}b")
