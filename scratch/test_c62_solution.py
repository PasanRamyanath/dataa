import os, sys, glob, json

sys.path.insert(0, os.path.abspath('feature engineering scripts'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from execution.champion_challenger_framework import Cycle35_ReversionPureScalper
from test_accurate_57_cycles import AccurateHistoricalDecision

class Cycle62_MidpointVetoScalper(Cycle35_ReversionPureScalper):
    """
    Candidate Solution for Cycle 62:
    Vetoes entering in the choppy 40%-60% interior of the 15m range unless backed by strong momentum.
    """
    def __init__(self, symbol="BTCUSDT"):
        super().__init__(symbol=symbol)
        self.midpoint_lower_pct = 0.40
        self.midpoint_upper_pct = 0.60

    def _is_midpoint_chop_zone(self, cand_dir: int, mid: float, decision) -> bool:
        if len(self.rolling_15m_prices) < 15:
            return False
        hi = max(self.rolling_15m_prices)
        lo = min(self.rolling_15m_prices)
        rng = hi - lo
        if rng < 10.0:
            return False
        pos_pct = (mid - lo) / rng

        # If price is in the 40%-60% dead-center zone:
        if self.midpoint_lower_pct <= pos_pct <= self.midpoint_upper_pct:
            # Only allow entry if momentum is exceptional (edge >= 35 bps and clear meso spread)
            d_meso = decision.meso_probs[2] - decision.meso_probs[0]
            if cand_dir == 0 and decision.expected_net_edge_bps >= 35.0 and d_meso <= -0.15:
                return False  # Allowed on high momentum breakdown
            if cand_dir == 2 and decision.expected_net_edge_bps >= 35.0 and d_meso >= 0.15:
                return False  # Allowed on high momentum breakout
            return True  # Veto midpoint chop
        return False

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        if self.state == "IDLE":
            mid = 0.5 * (bbo_bid + bbo_ask)
            if self._is_midpoint_chop_zone(decision.primary_direction, mid, decision):
                return self.state
        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)

ledger = "scratch/archive_cycle61/live_hierarchical_session_1h_ledger.jsonl"
records = []
with open(ledger, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

for name, cls in [("Reigning Champion Baseline", Cycle35_ReversionPureScalper),
                  ("Cycle 62 Midpoint Veto Candidate", Cycle62_MidpointVetoScalper)]:
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
