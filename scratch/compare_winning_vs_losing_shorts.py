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

winning_shorts = []
losing_shorts = []

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

        prev_pos = engine.active_position
        prev_closed_len = len(engine.closed_trades)

        engine.on_market_micro_tick(ts_ms=ts_ms, bbo_bid=bid, bbo_ask=ask, bbo_bid_qty=bid_qty, bbo_ask_qty=ask_qty, bar_high=ask+0.05, bar_low=bid-0.05, current_decision=decision)
        engine.on_hierarchical_decision(decision, bid, ask, bid_qty, ask_qty, 10000.0, mid)
        if engine.state == 'RESTING_ENTRY_QUEUE':
            engine.on_market_micro_tick(ts_ms=ts_ms+1000, bbo_bid=bid, bbo_ask=ask, bbo_bid_qty=bid_qty, bbo_ask_qty=ask_qty, bar_high=ask+0.05, bar_low=bid-0.05, current_decision=decision)

        if len(engine.closed_trades) > prev_closed_len:
            t = engine.closed_trades[-1]
            if t.side == 'SELL':
                # find entry bar
                ent_idx = max(0, idx - int(t.holding_duration_sec/10))
                ent_bar = bars[ent_idx]
                ent_mid = ent_bar.get('bbo_mid', ent_bar.get('mid'))
                ent_macro = ent_bar.get('macro_probs', [0.33, 0.33, 0.33])
                ent_meso = ent_bar.get('meso_probs', [0.33, 0.33, 0.33])
                ent_tact = ent_bar.get('tactical_probs', [0.33, 0.33, 0.33])
                ent_micro = ent_bar.get('micro_probs', [0.33, 0.33, 0.33])
                
                # session open drift
                sess_open = bars[0].get('bbo_mid', bars[0].get('mid'))
                drift_from_open = ((ent_mid - sess_open) / sess_open) * 1e4
                
                # 30-bar rolling return
                r_start = max(0, ent_idx - 30)
                r_start_mid = bars[r_start].get('bbo_mid', bars[r_start].get('mid'))
                ret_30 = ((ent_mid - r_start_mid) / r_start_mid) * 1e4
                
                # Range position
                mids_up_to_ent = [bars[k].get('bbo_mid', bars[k].get('mid')) for k in range(ent_idx + 1)]
                rng = max(mids_up_to_ent) - min(mids_up_to_ent)
                pct_rng = (ent_mid - min(mids_up_to_ent)) / rng if rng > 0 else 0.5

                item = {
                    'cycle': arch, 'net': t.net_return_bps, 'dur': t.holding_duration_sec,
                    'bar_idx': ent_idx, 'drift_open': drift_from_open, 'ret_30': ret_30,
                    'pct_rng': pct_rng, 'd_macro': ent_macro[2] - ent_macro[0],
                    'd_meso': ent_meso[2] - ent_meso[0], 'd_tact': ent_tact[2] - ent_tact[0],
                    'd_micro': ent_micro[2] - ent_micro[0], 'edge': ent_bar.get('expected_net_edge_bps', 0.0)
                }
                if t.net_return_bps > 0: winning_shorts.append(item)
                else: losing_shorts.append(item)

print(f"Total Winning Shorts: {len(winning_shorts)}")
print(f"Total Losing Shorts:  {len(losing_shorts)}")

print("\n--- LOSING SHORTS DETAIL ---")
for it in losing_shorts:
    print(f"  {it['cycle']}: Net={it['net']:+.1f}b | Bar={it['bar_idx']} | DriftOpen={it['drift_open']:+.1f}b | Ret30={it['ret_30']:+.1f}b | PctRng={it['pct_rng']*100:.1f}% | d_macro={it['d_macro']:+.3f} | d_meso={it['d_meso']:+.3f}")

print("\n--- WINNING SHORTS AVERAGES ---")
print(f"  Avg DriftOpen: {np.mean([x['drift_open'] for x in winning_shorts]):+.1f} bps (min={np.min([x['drift_open'] for x in winning_shorts]):+.1f}, max={np.max([x['drift_open'] for x in winning_shorts]):+.1f})")
print(f"  Avg Ret30:     {np.mean([x['ret_30'] for x in winning_shorts]):+.1f} bps (min={np.min([x['ret_30'] for x in winning_shorts]):+.1f}, max={np.max([x['ret_30'] for x in winning_shorts]):+.1f})")
print(f"  Avg PctRng:    {np.mean([x['pct_rng'] for x in winning_shorts])*100:.1f}% (min={np.min([x['pct_rng'] for x in winning_shorts])*100:.1f}%, max={np.max([x['pct_rng'] for x in winning_shorts])*100:.1f}%)")
print(f"  Avg d_macro:   {np.mean([x['d_macro'] for x in winning_shorts]):+.3f} (min={np.min([x['d_macro'] for x in winning_shorts]):+.3f}, max={np.max([x['d_macro'] for x in winning_shorts]):+.3f})")
print(f"  Avg d_meso:    {np.mean([x['d_meso'] for x in winning_shorts]):+.3f} (min={np.min([x['d_meso'] for x in winning_shorts]):+.3f}, max={np.max([x['d_meso'] for x in winning_shorts]):+.3f})")
