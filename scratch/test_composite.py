import os
import sys

sys.path.insert(0, os.path.abspath('feature engineering scripts'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from test_c63_challengers_benchmark import (
    Cycle63_PassiveMakerUnwindScalper,
    Cycle63_PostInvalidationSpacingScalper,
    replay_cycle_ledger,
)

benchmark_suite = [
    "archive_cycle62",
    "archive_cycle61",
    "archive_cycle60",
    "archive_cycle59",
    "archive_cycle58",
    "archive_cycle57",
    "archive_cycle35",
]


class Cycle63_CompositeOptimalScalper(Cycle63_PostInvalidationSpacingScalper):
    """
    Composite Optimal Scalper:
    Combines:
    1. Cycle 62 Winning Midpoint Chop Veto
    2. Post-Invalidation Retest Spacing (blocks whipsaw re-entries)
    3. Passive Maker Unwinds at BBO (saving 3.0 bps per invalidation cut)
    """
    def __init__(self, symbol: str = "BTCUSDT"):
        super().__init__(symbol=symbol)
        self.strategy_profile = "Challenger Composite Optimal Scalper"
        self.taker_fee_leg = 2.0  # Converted to Maker fee

    def _close_position(self, exit_price, exit_ts_ms, exit_fee_bps, reason="TAKE_PROFIT"):
        if "INVALIDATION" in reason:
            exit_fee_bps = self.maker_fee_leg
            reason = "PASSIVE_MAKER_UNWIND"
        return super()._close_position(exit_price, exit_ts_ms, exit_fee_bps, reason)


if __name__ == "__main__":
    tot_trades = 0
    tot_wins = 0
    tot_losses = 0
    tot_net_pnl = 0.0
    tot_fees = 0.0

    for c in benchmark_suite:
        ledger_path = os.path.join("scratch", c, "live_hierarchical_session_1h_ledger.jsonl")
        eng = Cycle63_CompositeOptimalScalper()
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
    print(f"Composite Optimal Scalper | Trades: {tot_trades:2d} | Wins: {tot_wins:2d} | Losses: {tot_losses:2d} | WR: {wr:5.1f}% | Net: {tot_net_pnl:+6.2f} bps | Fees: {tot_fees:5.1f}b")
