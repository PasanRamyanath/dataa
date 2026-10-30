import os
import glob
import json
import sys
import types
import numpy as np

sys.path.append('feature engineering scripts')
from inspect_the_5_short_losses_deep import BaselineTest

base_dir = 'scratch'
archives = sorted([d for d in os.listdir(base_dir) if (d.startswith('archive_cycle') or d.startswith('archive_v3')) and os.path.isdir(os.path.join(base_dir, d))])

win_short_durs = []
loss_short_durs = []

for arch in archives:
    ledgers = glob.glob(os.path.join(base_dir, arch, '*ledger*.jsonl'))
    if not ledgers: continue
    with open(ledgers[0], 'r', encoding='utf-8') as f:
        bars = [json.loads(l) for l in f if l.strip().startswith('{')]

    engine = BaselineTest()

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

    for t in engine.closed_trades:
        if t.side == 'SELL':
            if t.net_return_bps > 0:
                win_short_durs.append((t.holding_duration_sec, t.net_return_bps, t.exit_reason))
            else:
                loss_short_durs.append((t.holding_duration_sec, t.net_return_bps, t.exit_reason))

durs_only = [d for d, _, _ in win_short_durs]
print(f"Total Winning Short Trades: {len(win_short_durs)}")
print(f"Winning Short Holding Duration Distribution:")
print(f"  Median duration: {np.percentile(durs_only, 50):.1f}s")
print(f"  75th percentile: {np.percentile(durs_only, 75):.1f}s")
print(f"  90th percentile: {np.percentile(durs_only, 90):.1f}s")
print(f"  Max duration:    {max(durs_only):.1f}s")
print(f"\nWinning Shorts lasting > 600s: {len([d for d in durs_only if d > 600])}")
for d, net, r in win_short_durs:
    if d > 600:
        print(f"  Dur={d:.1f}s -> Net={net:+.2f} bps ({r})")

print(f"\nLosing Short Holding Durations:")
for d, net, r in loss_short_durs:
    print(f"  Dur={d:.1f}s -> Net={net:+.2f} bps ({r})")
