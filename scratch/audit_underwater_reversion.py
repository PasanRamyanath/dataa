import os
import glob
import json
import sys
import types

sys.path.append('feature engineering scripts')
from execution.champion_challenger_framework import Cycle65_UnifiedPeakReversionEngine

base_dir = 'scratch'
archives = sorted([d for d in os.listdir(base_dir) if (d.startswith('archive_cycle') or d.startswith('archive_v3')) and os.path.isdir(os.path.join(base_dir, d))])

underwater_at_300s = []

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

        pos = engine.active_position
        if pos is not None:
            elapsed = (ts_ms - pos.entry_ts_ms) / 1000.0
            sign = 1.0 if pos.side == "BUY" else -1.0
            unreal = sign * ((mid - pos.entry_price) / pos.entry_price) * 1e4
            if 290.0 <= elapsed <= 310.0 and not getattr(pos, '_logged_300s', False):
                pos._logged_300s = True
                underwater_at_300s.append({
                    'cycle': arch, 'side': pos.side, 'unreal_300s': unreal,
                    'mfe_300s': pos.mfe, 'mae_300s': pos.mae, 'entry_p': pos.entry_price
                })

        engine.on_market_micro_tick(ts_ms=ts_ms, bbo_bid=bid, bbo_ask=ask, bbo_bid_qty=bid_qty, bbo_ask_qty=ask_qty, bar_high=ask+0.05, bar_low=bid-0.05, current_decision=decision)
        engine.on_hierarchical_decision(decision, bid, ask, bid_qty, ask_qty, 10000.0, mid)
        if engine.state == 'RESTING_ENTRY_QUEUE':
            engine.on_market_micro_tick(ts_ms=ts_ms+1000, bbo_bid=bid, bbo_ask=ask, bbo_bid_qty=bid_qty, bbo_ask_qty=ask_qty, bar_high=ask+0.05, bar_low=bid-0.05, current_decision=decision)

    for item in underwater_at_300s:
        # find matching closed trade
        for t in engine.closed_trades:
            if t.side == item['side'] and abs(t.entry_price - item['entry_p']) < 0.1:
                item['final_net'] = t.net_return_bps
                item['final_reason'] = t.exit_reason
                break

print(f"Total trades active at t=300s: {len(underwater_at_300s)}")
print("\nTrades active at 300s breakdown:")
for it in underwater_at_300s:
    print(f"  {it['cycle']} {it['side']}: Unreal@300s={it['unreal_300s']:+.2f}b, MFE@300s={it['mfe_300s']:+.2f}b -> Final Net: {it.get('final_net', 0):+.2f}b ({it.get('final_reason')})")
