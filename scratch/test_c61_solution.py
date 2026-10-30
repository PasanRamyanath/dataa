import os, sys, glob, json

sys.path.insert(0, os.path.abspath('feature engineering scripts'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from execution.champion_challenger_framework import Cycle35_ReversionPureScalper

# Let's inspect what happens to Cycle 60 trades when we add a Re-Short Retrace Requirement
class Cycle61_PullbackExhaustionShortScalper(Cycle35_ReversionPureScalper):
    """
    Solves Cycle 60 Trade 5 flaw:
    Vetoes re-shorting below last exit price without a minimum pullback/retrace of >= 6.0 bps ($45+),
    preventing selling into the exact exhaustion wick of an oversold waterfall.
    """
    def __init__(self, symbol="BTCUSDT"):
        super().__init__(symbol=symbol)
        self.min_short_pullback_bps = 5.0  # Require >= 5.0 bps bounce before re-shorting

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        now_ms = decision.ts_ms
        mid = 0.5 * (bbo_bid + bbo_ask)

        if self.state == "IDLE" and decision.primary_direction == 0:
            # If last exit was a SHORT and we are IDLE:
            if self.last_exit_dir == 0 and self.last_exit_price > 0:
                # If price is lower than last exit price, require a pullback
                if mid < self.last_exit_price:
                    # If we haven't retraced up by at least min_short_pullback_bps from the recent low, veto entry!
                    if self.rolling_15m_prices:
                        recent_low = min(list(self.rolling_15m_prices)[-18:]) if len(self.rolling_15m_prices) >= 18 else min(self.rolling_15m_prices)
                        bounce_from_low_bps = ((mid - recent_low) / recent_low) * 1e4
                        if bounce_from_low_bps < self.min_short_pullback_bps:
                            return self.state

        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)

from test_accurate_57_cycles import AccurateHistoricalDecision

ledger = "scratch/archive_cycle60/live_hierarchical_session_1h_ledger.jsonl"
records = []
with open(ledger, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

# Compare Baseline Champion vs Cycle 61 Pullback Scalper on Cycle 60 ledger
for name, cls in [("Reigning Champion", Cycle35_ReversionPureScalper), ("Cycle 61 Candidate", Cycle61_PullbackExhaustionShortScalper)]:
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
