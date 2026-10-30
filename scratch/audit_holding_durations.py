import os
import glob
import json
import sys
import types

sys.path.append('feature engineering scripts')
from execution.champion_challenger_framework import Cycle65_UnifiedPeakReversionEngine

base_dir = 'scratch'
archives = sorted([d for d in os.listdir(base_dir) if (d.startswith('archive_cycle') or d.startswith('archive_v3')) and os.path.isdir(os.path.join(base_dir, d))])

trades_over_900s = []
trades_under_900s = []

for arch in archives:
    ledgers = glob.glob(os.path.join(base_dir, arch, '*ledger*.jsonl'))
    if not ledgers: continue
    with open(ledgers[0], 'r', encoding='utf-8') as f:
        bars = [json.loads(l) for l in f if l.strip().startswith('{')]

    engine = Cycle65_UnifiedPeakReversionEngine(symbol='BTCUSDT')

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
        item = {
            'cycle': arch, 'side': t.side, 'dur': t.holding_duration_sec,
            'net': t.net_return_bps, 'reason': t.exit_reason
        }
        if t.holding_duration_sec > 900.0:
            trades_over_900s.append(item)
        else:
            trades_under_900s.append(item)

print(f"Trades with duration <= 900s: {len(trades_under_900s)}")
w_under = sum(1 for t in trades_under_900s if t['net'] > 0)
net_under = sum(t['net'] for t in trades_under_900s)
print(f"  Under 900s: {w_under}W / {len(trades_under_900s)-w_under}L | WR: {w_under/len(trades_under_900s)*100:.1f}% | Net: {net_under:+.2f} bps")

print(f"\nTrades with duration > 900s: {len(trades_over_900s)}")
w_over = sum(1 for t in trades_over_900s if t['net'] > 0)
net_over = sum(t['net'] for t in trades_over_900s)
print(f"  Over 900s: {w_over}W / {len(trades_over_900s)-w_over}L | WR: {w_over/max(1, len(trades_over_900s))*100:.1f}% | Net: {net_over:+.2f} bps")

print("\nList of ALL Trades lasting > 900s:")
for t in trades_over_900s:
    print(f"  {t['cycle']} {t['side']}: Dur={t['dur']:.1f}s | Net={t['net']:+.2f} bps | Reason={t['reason']}")
