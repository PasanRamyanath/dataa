import os, sys, glob, json
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath('feature engineering scripts'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from execution.champion_challenger_framework import Cycle35_ReversionPureScalper

class AccurateHistoricalDecision:
    def __init__(self, d):
        self.ts_ms = int(d.get("ts_ms", 0))
        self.macro_probs = tuple(d.get("macro_probs", [0.33, 0.34, 0.33]))
        self.meso_probs = tuple(d.get("meso_probs", [0.33, 0.34, 0.33]))
        self.tactical_probs = tuple(d.get("tactical_probs", [0.33, 0.34, 0.33]))
        self.micro_probs = tuple(d.get("micro_probs", [0.33, 0.34, 0.33]))
        self.primary_direction = int(d.get("primary_direction", 1))
        self.expected_net_edge_bps = float(d.get("expected_net_edge_bps", 0.0))
        self.expected_gross_move_bps = self.expected_net_edge_bps + 4.0
        self.target_holding_seconds = 300
        self.take_profit_bps = 15.0
        self.stop_loss_bps = 25.0
        self.kelly_position_fraction = 0.10
        self.fused_swing_probs = self.meso_probs
        self.cross_horizon_alignment = float(self.meso_probs[self.primary_direction] - self.meso_probs[2 - self.primary_direction]) if self.primary_direction in (0, 2) else 0.0
        self.meta_act_probability = 0.65 if self.expected_net_edge_bps >= 8.0 else 0.35
        # ONLY trigger micro sniper if mode is PASSIVE_MAKER_SNIPER or URGENT_HYBRID
        self.execution_mode = d.get("execution_mode", "NONE")
        self.micro_sniper_trigger_prob = 0.75 if self.execution_mode in ("PASSIVE_MAKER_SNIPER", "URGENT_HYBRID") else 0.25

cycle_dirs = sorted([d for d in glob.glob("scratch/archive_cycle*") if os.path.isdir(d)],
                     key=lambda x: int(x.split("archive_cycle")[-1]) if x.split("archive_cycle")[-1].isdigit() else 999)

print(f"Testing Accurate Replayer on {len(cycle_dirs)} Cycle Directories...")

total_trades = 0
total_wins = 0
total_losses = 0
total_net_pnl = 0.0

for c_dir in cycle_dirs:
    ledger = os.path.join(c_dir, "live_hierarchical_session_1h_ledger.jsonl")
    if not os.path.exists(ledger):
        continue

    c_name = os.path.basename(c_dir)
    engine = Cycle35_ReversionPureScalper(symbol="BTCUSDT")

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

        # evaluate intra-bar high/low if present
        h = max(bid, ask, mid)
        l = min(bid, ask, mid)

        engine.on_market_micro_tick(
            ts_ms=ts_ms,
            bbo_bid=bid,
            bbo_ask=ask,
            bbo_bid_qty=1.0,
            bbo_ask_qty=1.0,
            bar_high=h,
            bar_low=l,
            current_decision=dec,
        )
        engine.on_hierarchical_decision(
            decision=dec,
            bbo_bid=bid,
            bbo_ask=ask,
            bbo_bid_qty=1.0,
            bbo_ask_qty=1.0,
            equity_usdt=10000.0,
            intra_bar_open=mid,
        )

    trades = engine.closed_trades
    if trades:
        c_wins = sum(1 for t in trades if t.net_return_bps > 0)
        c_pnl = sum(t.net_return_bps for t in trades)
        total_trades += len(trades)
        total_wins += c_wins
        total_losses += (len(trades) - c_wins)
        total_net_pnl += c_pnl
        print(f"[{c_name}] Trades: {len(trades)} | Wins: {c_wins}/{len(trades)} ({c_wins/len(trades)*100:.1f}%) | Net: {c_pnl:+.2f} bps")

print("=" * 60)
wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
print(f"OVERALL REPLAY: Trades: {total_trades} | Wins: {total_wins}/{total_trades} ({wr:.1f}%) | Total Net PnL: {total_net_pnl:+.2f} bps")
