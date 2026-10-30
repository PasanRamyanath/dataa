import os, sys, glob, json

sys.path.insert(0, os.path.abspath('feature engineering scripts'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from execution.champion_challenger_framework import Cycle35_ReversionPureScalper
from test_accurate_57_cycles import AccurateHistoricalDecision

class StrictRangeExtremeScalper(Cycle35_ReversionPureScalper):
    """
    Candidate: Strict Range Extreme Scalper.
    Only allows SHORT entries if price is in top 40% (pos_pct >= 0.60) of 15m range,
    and LONG entries if price is in bottom 40% (pos_pct <= 0.40) of 15m range,
    OR if edge >= 45 bps (parabolic breakout).
    """
    def __init__(self, symbol="BTCUSDT"):
        super().__init__(symbol=symbol)

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        if self.state == "IDLE":
            mid = 0.5 * (bbo_bid + bbo_ask)
            if len(self.rolling_15m_prices) >= 15:
                hi = max(self.rolling_15m_prices)
                lo = min(self.rolling_15m_prices)
                rng = hi - lo
                if rng >= 15.0:
                    pos_pct = (mid - lo) / rng
                    # For a SHORT, don't enter if price is in the lower 60% of the range unless massive edge:
                    if decision.primary_direction == 0 and pos_pct < 0.60 and decision.expected_net_edge_bps < 45.0:
                        return self.state
                    # For a LONG, don't enter if price is in the upper 60% of the range unless massive edge:
                    if decision.primary_direction == 2 and pos_pct > 0.40 and decision.expected_net_edge_bps < 45.0:
                        return self.state

        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)

ledger = "scratch/archive_cycle61/live_hierarchical_session_1h_ledger.jsonl"
records = []
with open(ledger, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

engine = StrictRangeExtremeScalper(symbol="BTCUSDT")
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
print(f"\n[Strict Range Extreme Scalper] Total Trades: {len(trades)} | Wins: {wins}/{len(trades)} ({wr:.1f}%) | Net: {pnl:+.2f} bps")
for i, t in enumerate(trades):
    print(f"  Trade {i+1}: {t.side} @ {t.entry_price:.1f} -> {t.exit_price:.1f} | Net: {t.net_return_bps:+.2f}b | Exit: {t.exit_reason}")
