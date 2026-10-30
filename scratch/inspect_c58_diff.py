import os, sys, glob, json

sys.path.insert(0, os.path.abspath('feature engineering scripts'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from execution.champion_challenger_framework import Cycle35_ReversionPureScalper
from test_accurate_57_cycles import AccurateHistoricalDecision

ledger = "scratch/archive_cycle58/live_hierarchical_session_1h_ledger.jsonl"
records = []
with open(ledger, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

engine = Cycle35_ReversionPureScalper(symbol="BTCUSDT")
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

print("CYCLE 58 REPLAY TRADES:")
for i, t in enumerate(engine.closed_trades):
    print(f"  Trade {i+1}: {t.side} @ {t.entry_price:.1f} -> {t.exit_price:.1f} | Net: {t.net_return_bps:+.2f}b | Exit: {t.exit_reason}")
