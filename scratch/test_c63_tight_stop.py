import os, sys, glob, json

sys.path.insert(0, os.path.abspath('feature engineering scripts'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from test_c62_solution import Cycle62_MidpointVetoScalper
from test_accurate_57_cycles import AccurateHistoricalDecision

class Cycle63_AsymmetricTightStopScalper(Cycle62_MidpointVetoScalper):
    """
    Candidate: Asymmetric Risk-Capped Midpoint Scalper.
    Combines:
    1. Midpoint Equilibrium Range Veto (C62 winner logic)
    2. Tightened Dual-Horizon Invalidation (-4.50 bps instead of -6.50 bps)
       to cap loss excursion before fees.
    """
    def __init__(self, symbol="BTCUSDT"):
        super().__init__(symbol=symbol)

    def _manage_active_swing_position(self, decision, bbo_bid, bbo_ask, now_ms):
        if self.active_position is None:
            return
        pos = self.active_position
        sign = 1.0 if pos.side == "BUY" else -1.0
        mid = 0.5 * (bbo_bid + bbo_ask)
        unrealized_bps = sign * ((mid - pos.entry_price) / pos.entry_price) * 1e4

        d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
        d_micro = decision.micro_probs[2] - decision.micro_probs[0]
        d_meso = decision.meso_probs[2] - decision.meso_probs[0]
        opp_dir = 0 if pos.side == "BUY" else 2

        # Fast Tight Invalidation at -4.50 bps (saves 2.0-3.0 bps per loss)
        if unrealized_bps <= -4.50:
            tact_opposing = (sign * d_tact) < -0.05
            micro_opposing = (sign * d_micro) < -0.05
            if tact_opposing or micro_opposing or decision.primary_direction == opp_dir:
                taker_exit_p = bbo_bid if pos.side == "BUY" else bbo_ask
                self._close_position(
                    exit_price=taker_exit_p,
                    exit_ts_ms=now_ms,
                    exit_fee_bps=self.taker_fee_leg,
                    reason="DUAL_HORIZON_INVALIDATION_TAKER",
                )
                return

        super()._manage_active_swing_position(decision, bbo_bid, bbo_ask, now_ms)

ledger = "scratch/archive_cycle62/live_hierarchical_session_1h_ledger.jsonl"
records = []
with open(ledger, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

for name, cls in [("Cycle 62 Midpoint Veto Leader", Cycle62_MidpointVetoScalper),
                  ("Cycle 63 Asymmetric Tight Stop Scalper", Cycle63_AsymmetricTightStopScalper)]:
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
