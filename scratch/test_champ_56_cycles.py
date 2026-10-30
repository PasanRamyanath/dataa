import sys, os, glob, json
sys.path.insert(0, os.path.abspath('feature engineering scripts'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from execution.champion_challenger_framework import Cycle35_ReversionPureScalper, HierarchicalDecisionOutput

class RealisticLedgerReplayer:
    def __init__(self, ledger_path: str):
        self.records = []
        with open(ledger_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))

    def replay(self, engine):
        engine.state = "IDLE"
        engine.active_position = None
        engine.closed_trades = []
        engine.ticks_seen = 0
        engine.rolling_15m_prices.clear()
        engine.recent_mid_prices.clear()
        engine.last_exit_ts_ms = 0
        engine.last_trade_exit_price = 0.0
        engine.last_trade_side = ""

        # Pre-seed 15 bars
        for r in self.records[:min(15, len(self.records))]:
            engine.rolling_15m_prices.append(float(r["bbo_mid"]))

        class MockDec:
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
                self.cross_horizon_alignment = 0.8
                self.meta_act_probability = 0.8
                self.micro_sniper_trigger_prob = 0.8
                self.execution_mode = str(d.get("execution_mode", "NONE"))

        for r in self.records:
            mid = float(r["bbo_mid"])
            bid = float(r["bbo_bid"])
            ask = float(r["bbo_ask"])
            ts_ms = int(r["ts_ms"])
            dec = MockDec(r)

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

import re
def get_cycle_num(path: str) -> int:
    m = re.search(r'archive_cycle(\d+)', path)
    return int(m.group(1)) if m else 999

all_ledgers = sorted(
    glob.glob(str(os.path.abspath("scratch/archive_cycle*/live_hierarchical_session_1h_ledger.jsonl"))),
    key=get_cycle_num
)

print(f"Replaying Champion (Cycle35_ReversionPureScalper) across {len(all_ledgers)} cycles with real execution_mode...")
champ_trades = []
for l in all_ledgers:
    c_num = get_cycle_num(l)
    replayer = RealisticLedgerReplayer(l)
    eng = Cycle35_ReversionPureScalper(symbol="BTCUSDT")
    trades = replayer.replay(eng)
    for t in trades:
        champ_trades.append((c_num, t))

total_trades = len(champ_trades)
wins = sum(1 for _, t in champ_trades if t.net_return_bps > 0)
losses = sum(1 for _, t in champ_trades if t.net_return_bps <= 0)
total_pnl = sum(t.net_return_bps for _, t in champ_trades)
wr = (wins / total_trades * 100) if total_trades > 0 else 0

print(f"Total Trades: {total_trades}")
print(f"Wins: {wins} | Losses: {losses} | Win Rate: {wr:.1f}%")
print(f"Total Net PnL: {total_pnl:+.2f} bps")
