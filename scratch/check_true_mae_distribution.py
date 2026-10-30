import os
import glob
import json
import sys
import types
import numpy as np

sys.path.append('feature engineering scripts')
from execution.champion_challenger_framework import Cycle65_UnifiedPeakReversionEngine

base_dir = 'scratch'
archives = sorted([d for d in os.listdir(base_dir) if (d.startswith('archive_cycle') or d.startswith('archive_v3')) and os.path.isdir(os.path.join(base_dir, d))])

win_maes = []
loss_maes = []

class FixedMAEEngine(Cycle65_UnifiedPeakReversionEngine):
    def _manage_active_swing_position(self, decision, bbo_bid, bbo_ask, now_ms):
        if self.active_position is None: return
        pos = self.active_position
        sign = 1.0 if pos.side == "BUY" else -1.0
        mid = 0.5 * (bbo_bid + bbo_ask)
        unrealized_bps = sign * ((mid - pos.entry_price) / pos.entry_price) * 1e4
        if not hasattr(pos, 'true_mae'):
            pos.true_mae = min(0.0, unrealized_bps)
            pos.true_mfe = max(0.0, unrealized_bps)
        else:
            pos.true_mae = min(pos.true_mae, unrealized_bps)
            pos.true_mfe = max(pos.true_mfe, unrealized_bps)
        super()._manage_active_swing_position(decision, bbo_bid, bbo_ask, now_ms)

    def _close_position(self, exit_price, exit_ts_ms, exit_fee_bps, reason="TAKE_PROFIT"):
        if self.active_position is not None:
            t_mae = getattr(self.active_position, 'true_mae', 0.0)
            t_mfe = getattr(self.active_position, 'true_mfe', 0.0)
            side = self.active_position.side
            sign = 1.0 if side == "BUY" else -1.0
            gross = sign * ((exit_price - self.active_position.entry_price) / self.active_position.entry_price) * 1e4
            net = gross - (self.active_position.maker_entry_fee_bps + exit_fee_bps)
            if net > 0:
                win_maes.append((t_mae, t_mfe, net, reason))
            else:
                loss_maes.append((t_mae, t_mfe, net, reason))
        super()._close_position(exit_price, exit_ts_ms, exit_fee_bps, reason)

for arch in archives:
    ledgers = glob.glob(os.path.join(base_dir, arch, '*ledger*.jsonl'))
    if not ledgers: continue
    with open(ledgers[0], 'r', encoding='utf-8') as f:
        bars = [json.loads(l) for l in f if l.strip().startswith('{')]

    engine = FixedMAEEngine(symbol='BTCUSDT')

    for idx, bar in enumerate(bars):
        mid = bar.get('bbo_mid', bar.get('mid', 0.0))
        bid = bar.get('bbo_bid', mid - 0.05)
        ask = bar.get('bbo_ask', mid + 0.05)
        bid_qty = bar.get('bbo_bid_qty', 10.0)
        ask_qty = bar.get('bbo_ask_qty', 10.0)
        ts_ms = bar.get('ts_ms', idx * 10000)

        tact = bar.get('tactical_probs', [0.33, 0.33, 0.33])
        meso = bar.get('meso_probs', [0.33, 0.33, 0.33])
        macro = bar.get('macro_probs', [0.33, 0.33, 0.33])
        micro = bar.get('micro_probs', [0.33, 0.33, 0.33])
        p_dir = bar.get('primary_direction', bar.get('direction', 1))
        edge = bar.get('expected_net_edge_bps', bar.get('edge', 0.0))

        decision = types.SimpleNamespace(
            ts_ms=ts_ms, primary_direction=p_dir, expected_net_edge_bps=edge,
            tactical_probs=tact, meso_probs=meso, macro_probs=macro, micro_probs=micro,
            kelly_position_fraction=0.15, execution_mode='PASSIVE_MAKER_SNIPER',
            target_holding_seconds=900, take_profit_bps=35.0, stop_loss_bps=45.0,
            micro_sniper_trigger_prob=0.85
        )

        engine.on_market_micro_tick(ts_ms=ts_ms, bbo_bid=bid, bbo_ask=ask, bbo_bid_qty=bid_qty, bbo_ask_qty=ask_qty, bar_high=ask+0.05, bar_low=bid-0.05, current_decision=decision)
        engine.on_hierarchical_decision(decision, bid, ask, bid_qty, ask_qty, 10000.0, mid)
        if engine.state == 'RESTING_ENTRY_QUEUE':
            engine.on_market_micro_tick(ts_ms=ts_ms+1000, bbo_bid=bid, bbo_ask=ask, bbo_bid_qty=bid_qty, bbo_ask_qty=ask_qty, bar_high=ask+0.05, bar_low=bid-0.05, current_decision=decision)

maes_only = [m for m, _, _, _ in win_maes]
print(f"Total Winning Trades: {len(win_maes)}")
print(f"Total Losing Trades:  {len(loss_maes)}")
print(f"MAE Distribution of WINS:")
print(f"  Worst MAE of any win: {min(maes_only):.2f} bps")
print(f"  50th percentile: {np.percentile(maes_only, 50):.2f} bps")
print(f"  75th percentile: {np.percentile(maes_only, 75):.2f} bps")
print(f"  90th percentile: {np.percentile(maes_only, 90):.2f} bps")
print(f"  95th percentile: {np.percentile(maes_only, 95):.2f} bps")
print(f"  99th percentile: {np.percentile(maes_only, 99):.2f} bps")

print("\nAll Wins with MAE <= -10.0 bps:")
for m, mfe, net, r in win_maes:
    if m <= -10.0:
        print(f"  MAE={m:.2f} bps, MFE={mfe:.2f} bps -> Net={net:+.2f} bps ({r})")
