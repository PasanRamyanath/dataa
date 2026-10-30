import os, sys, glob, json

sys.path.insert(0, os.path.abspath('feature engineering scripts'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from execution.champion_challenger_framework import Cycle35_ReversionPureScalper
from test_c62_solution import Cycle62_MidpointVetoScalper
from test_accurate_57_cycles import AccurateHistoricalDecision

class Cycle63_PostInvalidationFilterScalper(Cycle62_MidpointVetoScalper):
    """
    Candidate Solution for Cycle 63:
    Combines:
    1. Midpoint Equilibrium Range Veto (Cycle 62 winner logic)
    2. Post-Invalidation High/Low Filter: After an invalidation stop-loss cut, forbids
       re-entering in the same direction within 120 seconds unless price has cleanly
       cleared the invalidation level or broken below recent support.
    """
    def __init__(self, symbol="BTCUSDT"):
        super().__init__(symbol=symbol)
        self.last_invalidation_ts_ms = 0
        self.last_invalidation_price = 0.0
        self.last_invalidation_dir = -1

    def _manage_active_swing_position(self, decision, bbo_bid, bbo_ask, now_ms):
        # Track invalidation exits
        prev_trades_count = len(self.closed_trades)
        super()._manage_active_swing_position(decision, bbo_bid, bbo_ask, now_ms)
        if len(self.closed_trades) > prev_trades_count:
            last_trade = self.closed_trades[-1]
            if "INVALIDATION" in str(last_trade.exit_reason):
                self.last_invalidation_ts_ms = now_ms
                self.last_invalidation_price = last_trade.exit_price
                self.last_invalidation_dir = 2 if last_trade.side == "BUY" else 0

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        if self.state == "IDLE":
            now_ms = decision.ts_ms
            mid = 0.5 * (bbo_bid + bbo_ask)

            # Check Post-Invalidation Filter
            if self.last_invalidation_ts_ms > 0:
                elapsed_sec = (now_ms - self.last_invalidation_ts_ms) / 1000.0
                if elapsed_sec < 120.0 and decision.primary_direction == self.last_invalidation_dir:
                    # If last trade was a SHORT invalidated by an upward bounce:
                    if self.last_invalidation_dir == 0:
                        # Veto re-shorting between the exit price and entry price (the chop zone)
                        if mid < self.last_invalidation_price and decision.expected_net_edge_bps < 35.0:
                            return self.state
                    # If last trade was a BUY invalidated by a downward drop:
                    elif self.last_invalidation_dir == 2:
                        if mid > self.last_invalidation_price and decision.expected_net_edge_bps < 35.0:
                            return self.state

        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)

ledger = "scratch/archive_cycle62/live_hierarchical_session_1h_ledger.jsonl"
records = []
with open(ledger, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

for name, cls in [("Reigning Champion Baseline", Cycle35_ReversionPureScalper),
                  ("Cycle 62 Midpoint Veto Leader", Cycle62_MidpointVetoScalper),
                  ("Cycle 63 Post-Invalidation Scalper", Cycle63_PostInvalidationFilterScalper)]:
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
