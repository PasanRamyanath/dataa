import os, sys, glob, json

sys.path.insert(0, os.path.abspath('feature engineering scripts'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from execution.champion_challenger_framework import Cycle35_ReversionPureScalper
from test_accurate_57_cycles import AccurateHistoricalDecision

# Candidate 1: High Edge Expansion Scalper (Requires >= 20.0 bps edge)
class Cycle62_HighEdgeExpansionScalper(Cycle35_ReversionPureScalper):
    """Requires >= 20.0 bps edge to eliminate marginal midpoint chops."""
    def __init__(self, symbol="BTCUSDT"):
        super().__init__(symbol=symbol)
        self.min_net_edge_bps = 20.0

# Candidate 2: Passive Unwind Maker Guard Scalper
class Cycle62_PassiveUnwindMakerGuardScalper(Cycle35_ReversionPureScalper):
    """Exits via BBO Maker instead of Taker penalty when invalidated."""
    def __init__(self, symbol="BTCUSDT"):
        super().__init__(symbol=symbol)

# Candidate 3: Micro-Momentum Alignment Scalper (C60 100% WR)
class Cycle62_PureMicroMomentumScalper(Cycle35_ReversionPureScalper):
    """Vetoes adverse 10s micro-momentum bursts (> 0.08)."""
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

# Candidate 4: Stepped Ratchet Scalper
class Cycle62_SteppedRatchetScalper(Cycle62_PureMicroMomentumScalper):
    """Locks in +6b net at +12b MFE."""
    def __init__(self, symbol="BTCUSDT"):
        super().__init__(symbol=symbol)
        self.ratchet_threshold_bps = 12.0
        self.ratchet_lock_bps = 6.0

# Candidate 5: Adaptive Range Boundary Scalper
class Cycle62_AdaptiveRangeBoundaryScalper(Cycle62_PureMicroMomentumScalper):
    """Vetoes entries in 40%-60% midpoint chop zone unless edge >= 35 bps."""
    def __init__(self, symbol="BTCUSDT"):
        super().__init__(symbol=symbol)
        self.midpoint_lower_pct = 0.40
        self.midpoint_upper_pct = 0.60

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        if self.state == "IDLE":
            mid = 0.5 * (bbo_bid + bbo_ask)
            if len(self.rolling_15m_prices) >= 15:
                hi = max(self.rolling_15m_prices)
                lo = min(self.rolling_15m_prices)
                rng = hi - lo
                if rng >= 10.0:
                    pos_pct = (mid - lo) / rng
                    if self.midpoint_lower_pct <= pos_pct <= self.midpoint_upper_pct and decision.expected_net_edge_bps < 35.0:
                        return self.state
        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)

candidates = [
    ("Champion Benchmark", Cycle35_ReversionPureScalper),
    ("Challenger 1 (High Edge Expansion)", Cycle62_HighEdgeExpansionScalper),
    ("Challenger 2 (Passive Unwind Maker Guard)", Cycle62_PassiveUnwindMakerGuardScalper),
    ("Challenger 3 (Pure Micro-Momentum)", Cycle62_PureMicroMomentumScalper),
    ("Challenger 4 (Stepped Ratchet Micro)", Cycle62_SteppedRatchetScalper),
    ("Challenger 5 (Adaptive Range Boundary)", Cycle62_AdaptiveRangeBoundaryScalper),
]

for c_num in [60, 61]:
    print(f"\n=================== CYCLE {c_num} HISTORICAL AUDIT ===================")
    ledger = f"scratch/archive_cycle{c_num}/live_hierarchical_session_1h_ledger.jsonl"
    records = []
    with open(ledger, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    for name, cls in candidates:
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
        print(f"  * {name:<42}: Trades: {len(trades)} | Wins: {wins}/{len(trades)} ({wr:5.1f}%) | Net: {pnl:+6.2f} bps")
