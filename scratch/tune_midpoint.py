import os, sys, glob, json

sys.path.insert(0, os.path.abspath('feature engineering scripts'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from test_c62_solution import Cycle62_MidpointVetoScalper
from test_accurate_57_cycles import AccurateHistoricalDecision

for lower, upper in [(0.40, 0.60), (0.35, 0.65), (0.30, 0.70), (0.25, 0.75)]:
    class TunedMidpointScalper(Cycle62_MidpointVetoScalper):
        def __init__(self, symbol="BTCUSDT"):
            super().__init__(symbol=symbol)
            self.midpoint_lower_pct = lower
            self.midpoint_upper_pct = upper

    ledger = "scratch/archive_cycle61/live_hierarchical_session_1h_ledger.jsonl"
    records = []
    with open(ledger, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    engine = TunedMidpointScalper(symbol="BTCUSDT")
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
    print(f"Zone [{lower*100:.0f}% - {upper*100:.0f}%]: Trades: {len(trades)} | Wins: {wins}/{len(trades)} ({wr:.1f}%) | Net: {pnl:+.2f} bps")
