import os, sys, glob, json

sys.path.insert(0, os.path.abspath('feature engineering scripts'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from execution.champion_challenger_framework import (
    Cycle35_ReversionPureScalper,
    Cycle35_TightRangeReversionScalper,
)
from test_accurate_57_cycles import AccurateHistoricalDecision

# Challenger 1: Micro-Momentum Alignment Scalper
class Cycle61_MicroMomentumAlignmentScalper(Cycle35_ReversionPureScalper):
    """
    Challenger 1: Micro-Momentum Filtered Scalper.
    Solves Cycle 60 Trade 5 loss: Vetoes entering SHORT when micro_probs[2] - micro_probs[0] > 0.08,
    and LONG when micro_probs[0] - micro_probs[2] > 0.08.
    """
    def __init__(self, symbol="BTCUSDT"):
        super().__init__(symbol=symbol)
        self.adverse_micro_threshold = 0.08

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        if self.state == "IDLE":
            d_micro_up = decision.micro_probs[2] - decision.micro_probs[0]
            if decision.primary_direction == 0 and d_micro_up > self.adverse_micro_threshold:
                return self.state
            if decision.primary_direction == 2 and (-d_micro_up) > self.adverse_micro_threshold:
                return self.state
        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)


# Challenger 2: Retrace Pullback Short Scalper
class Cycle61_RetracePullbackShortScalper(Cycle35_ReversionPureScalper):
    """
    Challenger 2: Retrace Pullback Scalper.
    Solves bottom waterfall chasing: When re-shorting below last trade exit,
    requires at least 4.0 bps retrace upward from recent 10-bar low before re-shorting.
    """
    def __init__(self, symbol="BTCUSDT"):
        super().__init__(symbol=symbol)
        self.min_short_retrace_bps = 4.0

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        if self.state == "IDLE" and decision.primary_direction == 0 and self.last_exit_dir == 0 and self.last_exit_price > 0:
            mid = 0.5 * (bbo_bid + bbo_ask)
            if mid < self.last_exit_price and len(self.recent_mid_prices) >= 10:
                recent_low = min(p for _, p in list(self.recent_mid_prices)[-10:])
                bounce_bps = ((mid - recent_low) / recent_low) * 1e4
                if bounce_bps < self.min_short_retrace_bps:
                    return self.state
        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)


# Challenger 3: Stepped Ratchet Scalper
class Cycle61_SteppedRatchetScalper(Cycle61_MicroMomentumAlignmentScalper):
    """
    Challenger 3: Stepped Ratchet + Micro-Momentum Scalper.
    Protects gains: at +12.0 bps MFE, locks in +6.0 bps net exit barrier.
    """
    def __init__(self, symbol="BTCUSDT"):
        super().__init__(symbol=symbol)
        self.ratchet_threshold_bps = 12.0
        self.ratchet_lock_bps = 6.0


# Challenger 4: Tight Range Envelope Scalper
class Cycle61_TightEnvelopeMicroScalper(Cycle61_MicroMomentumAlignmentScalper):
    """
    Challenger 4: Tight Range Envelope ($6) + Micro-Momentum Scalper.
    Confines entries to outer $6 range extremes while vetoing adverse micro-momentum.
    """
    def __init__(self, symbol="BTCUSDT"):
        super().__init__(symbol=symbol)
        self.range_boundary_buffer_dollars = 6.0


# Challenger 5: Dual Horizon Coherence Scalper
class Cycle61_DualHorizonCoherenceScalper(Cycle35_ReversionPureScalper):
    """
    Challenger 5: Dual Horizon Coherence Scalper.
    Requires Tactical and Meso direction to agree on primary direction (spread > 0.05 on both tiers).
    """
    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        if self.state == "IDLE":
            d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
            d_meso = decision.meso_probs[2] - decision.meso_probs[0]
            if decision.primary_direction == 0 and not (d_tact < -0.05 and d_meso < -0.05):
                return self.state
            if decision.primary_direction == 2 and not (d_tact > 0.05 and d_meso > 0.05):
                return self.state
        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)


# Evaluate on Cycle 60
ledger = "scratch/archive_cycle60/live_hierarchical_session_1h_ledger.jsonl"
records = []
with open(ledger, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

lineup = [
    ("Champion Benchmark", Cycle35_ReversionPureScalper),
    ("Challenger 1 (Micro-Momentum Alignment)", Cycle61_MicroMomentumAlignmentScalper),
    ("Challenger 2 (Retrace Pullback Short)", Cycle61_RetracePullbackShortScalper),
    ("Challenger 3 (Stepped Ratchet Micro)", Cycle61_SteppedRatchetScalper),
    ("Challenger 4 (Tight Envelope Micro)", Cycle61_TightEnvelopeMicroScalper),
    ("Challenger 5 (Dual Horizon Coherence)", Cycle61_DualHorizonCoherenceScalper),
]

print("CYCLE 60 REPLAY VALIDATION:")
for name, cls in lineup:
    engine = cls(symbol="BTCUSDT")
    for r in records:
        mid = float(r.get("bbo_mid", 0.0))
        bid = float(r.get("bbo_bid", mid))
        ask = float(r.get("bbo_ask", mid))
        ts_ms = int(r.get("ts_ms", 0))
        dec = AccurateHistoricalDecision(r)
        h = max(bid, ask, mid)
        l = min(bid, ask, mid)
        engine.on_market_micro_tick(ts_ms=ts_ms, bbo_bid=bid, bbo_ask=ask, bbo_bid_qty=1.0, bbo_ask_qty=1.0, bar_high=h, bar_low=l, current_decision=dec)
        engine.on_hierarchical_decision(decision=dec, bbo_bid=bid, bbo_ask=ask, bbo_bid_qty=1.0, bbo_ask_qty=1.0, equity_usdt=10000.0, intra_bar_open=mid)

    trades = engine.closed_trades
    wins = sum(1 for t in trades if t.net_return_bps > 0)
    pnl = sum(t.net_return_bps for t in trades)
    wr = (wins / len(trades) * 100) if trades else 0.0
    print(f"  * {name:<40}: Trades: {len(trades)} | Wins: {wins}/{len(trades)} ({wr:5.1f}%) | Net: {pnl:+6.2f} bps")
