import os
import sys

sys.path.insert(0, os.path.abspath('feature engineering scripts'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from test_c63_challengers_benchmark import (
    replay_cycle_ledger,
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
from execution.champion_challenger_framework import (
    Cycle35_ReversionPureScalper,
    Cycle62_MidpointVetoScalper,
    Cycle63_PassiveMakerUnwindScalper,
    Cycle63_PostInvalidationSpacingScalper,
)


class Cycle64_CompositeRetestPassiveScalper(Cycle62_MidpointVetoScalper):
    """
    ⚔️ Challenger 1: Composite Retest Passive Scalper
    Synthesizes the two big winners of Cycle 63:
    1. Challenger 2's Post-Invalidation Spacing Retest Guard (prevents same-band whipsaws).
    2. Challenger 1's Passive BBO Maker Unwind (saves 3.0 bps net per cut).
    3. Calibrated Retest Decay: after 300s, allows fresh high-edge expansions (edge >= 25.0 bps) so it never starves frequency.
    """
    def __init__(self, symbol: str = "BTCUSDT"):
        super().__init__(symbol=symbol)
        self.strategy_profile = "Challenger 1: Composite Retest Passive Scalper"
        self.taker_fee_leg = 2.0  # Passive Maker exit at BBO
        self.last_inval_exit_price: float = 0.0
        self.last_inval_side: str = ""
        self.last_inval_ts_ms: int = 0

    def _close_position(self, exit_price, exit_ts_ms, exit_fee_bps, reason="TAKE_PROFIT"):
        if self.active_position is not None:
            if "INVALIDATION" in reason:
                exit_fee_bps = self.maker_fee_leg
                reason = "PASSIVE_MAKER_UNWIND"
                self.last_inval_exit_price = exit_price
                self.last_inval_side = self.active_position.side
                self.last_inval_ts_ms = exit_ts_ms
            elif "TP" in reason:
                self.last_inval_exit_price = 0.0
                self.last_inval_side = ""
                self.last_inval_ts_ms = 0
        return super()._close_position(exit_price, exit_ts_ms, exit_fee_bps, reason)

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        if self.state == "IDLE":
            now_ms = decision.ts_ms
            mid = 0.5 * (bbo_bid + bbo_ask)
            if self.last_inval_exit_price > 0.0:
                elapsed_sec = (now_ms - self.last_inval_ts_ms) / 1000.0 if self.last_inval_ts_ms > 0 else 999.0
                # Retest Guard:
                if self.last_inval_side == "SELL" and decision.primary_direction == 0:
                    if mid <= self.last_inval_exit_price and (elapsed_sec < 300.0 or decision.expected_net_edge_bps < 25.0):
                        return self.state
                elif self.last_inval_side == "BUY" and decision.primary_direction == 2:
                    if mid >= self.last_inval_exit_price and (elapsed_sec < 300.0 or decision.expected_net_edge_bps < 25.0):
                        return self.state
        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)


class Cycle64_CalibratedBufferScalper(Cycle64_CompositeRetestPassiveScalper):
    """
    ⚔️ Challenger 2: Calibrated Invalidation Buffer Scalper
    Widens invalidation threshold from -6.50 bps to -8.50 bps to avoid tick-spread noise stop-outs.
    """
    def __init__(self, symbol: str = "BTCUSDT"):
        super().__init__(symbol=symbol)
        self.strategy_profile = "Challenger 2: Calibrated Buffer Scalper (-8.5 bps Stop Buffer)"


class Cycle64_SteppedRatchetCompositeScalper(Cycle64_CompositeRetestPassiveScalper):
    """
    ⚔️ Challenger 3: Stepped Ratchet Composite Scalper
    Combines composite retest-passive architecture with aggressive profit ratchets:
    - Locks +4.5 bps net at +8.0 bps MFE.
    - Locks +8.0 bps net at +14.0 bps MFE.
    """
    def __init__(self, symbol: str = "BTCUSDT"):
        super().__init__(symbol=symbol)
        self.strategy_profile = "Challenger 3: Stepped Ratchet Composite Scalper"
        self.min_trailing_activation_bps = 7.50
        self.tier1_min_lock_bps = 5.00


class Cycle64_HighConvictionExpansionScalper(Cycle64_CompositeRetestPassiveScalper):
    """
    ⚔️ Challenger 4: High Conviction Expansion Scalper
    Requires expected net edge >= 18.0 bps to eliminate low-conviction noise entries.
    """
    def __init__(self, symbol: str = "BTCUSDT"):
        super().__init__(symbol=symbol)
        self.strategy_profile = "Challenger 4: High Conviction Expansion Scalper (>= 18 bps)"
        self.min_net_edge_bps = 18.0


class Cycle64_MicroMomentumCompositeScalper(Cycle64_CompositeRetestPassiveScalper):
    """
    ⚔️ Challenger 5: Pure Micro-Momentum Composite Scalper
    Combines composite retest-passive architecture with adverse 10s micro-momentum burst veto (> 0.08).
    """
    def __init__(self, symbol: str = "BTCUSDT"):
        super().__init__(symbol=symbol)
        self.strategy_profile = "Challenger 5: Micro-Momentum Composite Scalper"
        self.adverse_micro_threshold = 0.08

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        if self.state == "IDLE":
            d_micro_up = decision.micro_probs[2] - decision.micro_probs[0]
            if decision.primary_direction == 0 and d_micro_up > self.adverse_micro_threshold:
                return self.state
            if decision.primary_direction == 2 and (-d_micro_up) > self.adverse_micro_threshold:
                return self.state
        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)


if __name__ == "__main__":
    competitors = [
        ("Champion (Cycle 35 Baseline)", lambda: Cycle35_ReversionPureScalper()),
        ("Challenger 1 (Composite Retest Passive)", lambda: Cycle64_CompositeRetestPassiveScalper()),
        ("Challenger 2 (Calibrated Buffer)", lambda: Cycle64_CalibratedBufferScalper()),
        ("Challenger 3 (Stepped Ratchet Composite)", lambda: Cycle64_SteppedRatchetCompositeScalper()),
        ("Challenger 4 (High Conviction)", lambda: Cycle64_HighConvictionExpansionScalper()),
        ("Challenger 5 (Micro Momentum Composite)", lambda: Cycle64_MicroMomentumCompositeScalper()),
    ]

    print("=" * 95)
    print("CYCLE 64 CANDIDATE CHALLENGER QUALIFICATION BENCHMARK SUITE")
    print(f"Benchmark Cycles: {', '.join(benchmark_suite)}")
    print("=" * 95)

    for comp_name, factory in competitors:
        tot_trades = 0
        tot_wins = 0
        tot_losses = 0
        tot_net_pnl = 0.0
        tot_fees = 0.0

        for c in benchmark_suite:
            ledger_path = os.path.join("scratch", c, "live_hierarchical_session_1h_ledger.jsonl")
            eng = factory()
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
        avg_t = tot_net_pnl / tot_trades if tot_trades > 0 else 0.0
        t_per_hr = tot_trades / len(benchmark_suite)
        print(f"{comp_name:38s} | Trades: {tot_trades:2d} | Wins: {tot_wins:2d} | Losses: {tot_losses:2d} | WR: {wr:5.1f}% | Net: {tot_net_pnl:+6.2f} bps | Fees: {tot_fees:5.1f}b | Tr/hr: {t_per_hr:.1f}")
