import os, sys, glob, json

sys.path.insert(0, os.path.abspath('feature engineering scripts'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from execution.champion_challenger_framework import Cycle35_ReversionPureScalper
from test_accurate_57_cycles import AccurateHistoricalDecision

class Cycle62_TacticalMesoCoherenceScalper(Cycle35_ReversionPureScalper):
    """
    Candidate: Tactical-Meso Coherence Scalper.
    Enforces that Tactical (1m-5m) MUST NOT conflict with primary direction:
    - For SHORT (0): Vetoes entry if tactical_probs[2] >= tactical_probs[0] (Tactical is Bullish/Flat)
    - For LONG (2): Vetoes entry if tactical_probs[0] >= tactical_probs[2] (Tactical is Bearish/Flat)
    """
    def __init__(self, symbol="BTCUSDT"):
        super().__init__(symbol=symbol)

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        if self.state == "IDLE":
            # Check Tactical Direction Alignment
            if decision.primary_direction == 0 and (decision.tactical_probs[2] >= decision.tactical_probs[0]):
                return self.state
            if decision.primary_direction == 2 and (decision.tactical_probs[0] >= decision.tactical_probs[2]):
                return self.state

        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)

ledger = "scratch/archive_cycle61/live_hierarchical_session_1h_ledger.jsonl"
records = []
with open(ledger, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

for name, cls in [("Baseline Champion", Cycle35_ReversionPureScalper),
                  ("Cycle 62 Tactical-Meso Coherence Candidate", Cycle62_TacticalMesoCoherenceScalper)]:
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
    print(f"\n[{name}] Total Trades: {len(trades)} | Wins: {wins}/{len(trades)} ({wr:.1f}%) | Net: {pnl:+.2f} bps")
    for i, t in enumerate(trades):
        print(f"  Trade {i+1}: {t.side} @ {t.entry_price:.1f} -> {t.exit_price:.1f} | Net: {t.net_return_bps:+.2f}b | Exit: {t.exit_reason}")
