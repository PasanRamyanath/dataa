import sys, os, json
sys.path.insert(0, os.path.abspath('feature engineering scripts'))
sys.path.insert(0, os.path.abspath('.'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from scratch.test_high_fidelity_replayer_v2 import HighFidelityReplayerV2
from execution.champion_challenger_framework import Cycle35_ReversionPureScalper

class CalibratedPureScalper(Cycle35_ReversionPureScalper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Calibrated Pure Scalper"

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        now_ms = decision.ts_ms
        mid = 0.5 * (bbo_bid + bbo_ask)

        if self.state == "IDLE":
            # Strict same-direction wave cooldown: 360s (6 min) after an exit before entering the same direction
            if self.last_exit_ts_ms > 0:
                elapsed = (now_ms - self.last_exit_ts_ms) / 1000.0
                cand_dir = decision.primary_direction
                if elapsed < 360.0 and self.last_trade_exit_price > 0:
                    if cand_dir == 2 and self.last_trade_side == "BUY" and (mid > self.last_trade_exit_price - 12.0):
                        return self.state
                    elif cand_dir == 0 and self.last_trade_side == "SELL" and (mid < self.last_trade_exit_price + 12.0):
                        return self.state

        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)

path12 = "scratch/archive_cycle12/live_hierarchical_session_1h_ledger.jsonl"
replayer = HighFidelityReplayerV2(path12)
eng = CalibratedPureScalper(symbol="BTCUSDT")
trades = replayer.replay_engine(eng)
wins = [t for t in trades if t.net_return_bps > 0]
pnl = sum(t.net_return_bps for t in trades)
wr = len(wins) / len(trades) * 100.0 if trades else 0.0
print(f"CYCLE 12 with Same-Direction Cooldown:")
print(f"Trades: {len(trades)} | Wins: {len(wins)} | WR: {wr:.1f}% | Net PnL: {pnl:+.2f} bps")
for j, t in enumerate(trades):
    print(f"  Trade {j+1}: {t.side} @ {t.entry_price:.1f} -> {t.exit_price:.1f} | Net: {t.net_return_bps:+.2f} bps | {t.exit_reason}")
