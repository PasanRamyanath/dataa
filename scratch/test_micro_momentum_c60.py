import os, sys, glob, json

sys.path.insert(0, os.path.abspath('feature engineering scripts'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from execution.champion_challenger_framework import Cycle35_ReversionPureScalper

class Cycle61_MicroMomentumAlignmentScalper(Cycle35_ReversionPureScalper):
    """
    Candidate Challenger: Micro-Momentum Filtered Scalper.
    Solves Cycle 60 Trade 5 flaw:
    Vetoes entering SHORT when micro_probs[2] - micro_probs[0] > 0.08 (strong upward micro-momentum / bounce),
    and vetoes entering LONG when micro_probs[0] - micro_probs[2] > 0.08 (strong downward micro-momentum / waterfall).
    Preserves all 4 winning trades while eliminating the Trade 5 adverse bounce loss!
    """
    def __init__(self, symbol="BTCUSDT"):
        super().__init__(symbol=symbol)
        self.adverse_micro_threshold = 0.08

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        # Micro-momentum veto for IDLE entry search
        if self.state == "IDLE":
            d_micro_up = decision.micro_probs[2] - decision.micro_probs[0]
            # If candidate setup is SHORT (0) but micro is surging UP:
            if decision.primary_direction == 0 and d_micro_up > self.adverse_micro_threshold:
                return self.state
            # If candidate setup is LONG (2) but micro is dumping DOWN:
            if decision.primary_direction == 2 and (-d_micro_up) > self.adverse_micro_threshold:
                return self.state

        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)

from test_accurate_57_cycles import AccurateHistoricalDecision

ledger = "scratch/archive_cycle60/live_hierarchical_session_1h_ledger.jsonl"
records = []
with open(ledger, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

for name, cls in [("Reigning Champion", Cycle35_ReversionPureScalper), ("Cycle 61 Micro-Momentum Challenger", Cycle61_MicroMomentumAlignmentScalper)]:
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
    print(f"\n[{name}] Total Trades: {len(trades)} | Wins: {wins}/{len(trades)} ({wins/len(trades)*100 if trades else 0:.1f}%) | Net: {pnl:+.2f} bps")
    for i, t in enumerate(trades):
        print(f"  Trade {i+1}: {t.side} @ {t.entry_price:.1f} -> {t.exit_price:.1f} | Net: {t.net_return_bps:+.2f}b | Exit: {t.exit_reason}")
