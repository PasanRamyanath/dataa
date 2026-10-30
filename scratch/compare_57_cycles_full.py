import os, sys, glob, json

sys.path.insert(0, os.path.abspath('feature engineering scripts'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from execution.champion_challenger_framework import Cycle35_ReversionPureScalper
from test_accurate_57_cycles import AccurateHistoricalDecision
from test_micro_momentum_c60 import Cycle61_MicroMomentumAlignmentScalper

cycle_dirs = sorted([d for d in glob.glob("scratch/archive_cycle*") if os.path.isdir(d)],
                     key=lambda x: int(x.split("archive_cycle")[-1]) if x.split("archive_cycle")[-1].isdigit() else 999)

print("=" * 80)
print(f"57-CYCLE HISTORICAL QUALIFICATION EVALUATION")
print("=" * 80)

for name, cls in [("Reigning Champion (Cycle 35 Baseline)", Cycle35_ReversionPureScalper),
                  ("Cycle 61 Candidate (Micro-Momentum Alignment Scalper)", Cycle61_MicroMomentumAlignmentScalper)]:
    total_trades = 0
    total_wins = 0
    total_net_pnl = 0.0
    cycles_profitable = 0
    total_cycles = 0

    for c_dir in cycle_dirs:
        ledger = os.path.join(c_dir, "live_hierarchical_session_1h_ledger.jsonl")
        if not os.path.exists(ledger):
            continue

        total_cycles += 1
        engine = cls(symbol="BTCUSDT")

        records = []
        with open(ledger, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

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
        if trades:
            c_wins = sum(1 for t in trades if t.net_return_bps > 0)
            c_pnl = sum(t.net_return_bps for t in trades)
            total_trades += len(trades)
            total_wins += c_wins
            total_net_pnl += c_pnl
            if c_pnl > 0:
                cycles_profitable += 1

    wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
    avg_per_hr = total_trades / total_cycles if total_cycles > 0 else 0
    print(f"\n[{name}]")
    print(f"  Total Cycles Evaluated: {total_cycles}")
    print(f"  Total Trades: {total_trades} (Avg {avg_per_hr:.2f} tr/hr)")
    print(f"  Win Rate: {total_wins}/{total_trades} ({wr:.1f}%)")
    print(f"  Net PnL (Replay): {total_net_pnl:+.2f} bps")
    print(f"  Profitable Cycles: {cycles_profitable}/{total_cycles} ({cycles_profitable/total_cycles*100:.1f}%)")
