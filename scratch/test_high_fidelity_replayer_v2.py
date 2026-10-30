import sys, os, json
sys.path.insert(0, os.path.abspath('feature engineering scripts'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from execution.champion_challenger_framework import Cycle35_ReversionPureScalper

class HighFidelityReplayerV2:
    def __init__(self, ledger_path: str):
        self.ledger_path = ledger_path
        self.records = []
        with open(ledger_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))

    def replay_engine(self, engine):
        engine.state = "IDLE"
        engine.active_position = None
        engine.closed_trades = []
        engine.ticks_seen = 0
        engine.rolling_15m_prices.clear()
        engine.recent_mid_prices.clear()
        engine.last_exit_ts_ms = 0
        engine.last_trade_exit_price = 0.0
        engine.last_trade_side = ""

        # Pre-seed first 15 bars
        seed_bars = min(15, len(self.records))
        for r in self.records[:seed_bars]:
            engine.rolling_15m_prices.append(float(r["bbo_mid"]))

        class MockDec:
            def __init__(self, d):
                self.ts_ms = int(d.get("ts_ms", 0))
                self.macro_probs = d.get("macro_probs", [0.33, 0.34, 0.33])
                self.meso_probs = d.get("meso_probs", [0.33, 0.34, 0.33])
                self.tactical_probs = d.get("tactical_probs", [0.33, 0.34, 0.33])
                self.micro_probs = d.get("micro_probs", [0.33, 0.34, 0.33])
                self.primary_direction = int(d.get("primary_direction", 1))
                self.expected_net_edge_bps = float(d.get("expected_net_edge_bps", 0.0))
                self.expected_gross_move_bps = self.expected_net_edge_bps + 4.0
                self.target_holding_seconds = 300.0
                self.take_profit_bps = 15.0
                self.stop_loss_bps = 25.0
                self.kelly_position_fraction = 0.10
                self.fused_swing_probs = self.meso_probs
                self.cross_horizon_alignment = 0.8
                self.meta_act_probability = 0.8
                self.micro_sniper_trigger_prob = 0.8
                self.execution_mode = "SNIPER_MAKER"

        for i, r in enumerate(self.records):
            mid = float(r["bbo_mid"])
            bid = float(r["bbo_bid"])
            ask = float(r["bbo_ask"])
            ts_ms = int(r["ts_ms"])
            dec = MockDec(r)

            # In live runner: micro tick is called first with current_decision!
            engine.on_market_micro_tick(
                ts_ms=ts_ms,
                bbo_bid=bid,
                bbo_ask=ask,
                bbo_bid_qty=1.0,
                bbo_ask_qty=1.0,
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

        return list(engine.closed_trades)

test_cycles = ["archive_cycle12", "archive_cycle35", "archive_cycle57", "archive_cycle59"]
print("=" * 80)
print("TESTING HIGH-FIDELITY REPLAYER V2 (WITH CURRENT DECISION CANCELLATION)")
print("=" * 80)

for c in test_cycles:
    path = f"scratch/{c}/live_hierarchical_session_1h_ledger.jsonl"
    replayer = HighFidelityReplayerV2(path)
    eng = Cycle35_ReversionPureScalper(symbol="BTCUSDT")
    trades = replayer.replay_engine(eng)
    wins = [t for t in trades if t.net_return_bps > 0]
    pnl = sum(t.net_return_bps for t in trades)
    wr = len(wins) / len(trades) * 100.0 if trades else 0.0
    print(f"\n[{c.replace('archive_', '').upper()}]: {len(trades)} trades | {len(wins)}W/{len(trades)-len(wins)}L ({wr:.1f}%) | Net: {pnl:+.2f} bps")
    for j, t in enumerate(trades):
        print(f"  Trade {j+1}: {t.side} @ {t.entry_price:.1f} -> {t.exit_price:.1f} | Net: {t.net_return_bps:+.2f} bps | {t.exit_reason}")
