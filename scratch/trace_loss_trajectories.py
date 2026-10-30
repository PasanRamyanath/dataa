import os
import glob
import json
import sys
import types

sys.path.append('feature engineering scripts')
from execution.champion_challenger_framework import Cycle65_UnifiedPeakReversionEngine

loss_cycles = ['archive_cycle17', 'archive_cycle40', 'archive_cycle43', 'archive_cycle52', 'archive_cycle57', 'archive_cycle58']
base_dir = 'scratch'

for cycle_name in loss_cycles:
    path = os.path.join(base_dir, cycle_name)
    ledgers = glob.glob(os.path.join(path, '*ledger*.jsonl'))
    if not ledgers: 
        continue
    with open(ledgers[0], 'r', encoding='utf-8') as f:
        bars = [json.loads(l) for l in f if l.strip().startswith('{')]
        
    engine = Cycle65_UnifiedPeakReversionEngine(symbol='BTCUSDT')
    print(f"\n==================== Trajectory for {cycle_name} ====================")
    
    in_loss_trade = False
    trade_start_idx = 0
    trade_side = ""
    entry_p = 0.0
    
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
        
        dec = types.SimpleNamespace(
            ts_ms=ts_ms, primary_direction=p_dir, expected_net_edge_bps=edge,
            tactical_probs=tact, meso_probs=meso, macro_probs=macro, micro_probs=micro,
            kelly_position_fraction=0.15, execution_mode='PASSIVE_MAKER_SNIPER',
            target_holding_seconds=900, take_profit_bps=35.0, stop_loss_bps=45.0,
            micro_sniper_trigger_prob=0.85
        )
        
        prev_pos = engine.active_position
        engine.on_market_micro_tick(ts_ms, bid, ask, bid_qty, ask_qty, ask + 0.05, bid - 0.05, dec)
        engine.on_hierarchical_decision(dec, bid, ask, bid_qty, ask_qty, 10000.0, mid)
        if engine.state == 'RESTING_ENTRY_QUEUE':
            engine.on_market_micro_tick(ts_ms + 1000, bid, ask, bid_qty, ask_qty, ask + 0.05, bid - 0.05, dec)
            
        pos = engine.active_position
        if pos is not None and prev_pos is None:
            # Just entered
            entry_p = pos.entry_price
            trade_side = pos.side
            trade_start_idx = idx
            print(f"ENTERED {trade_side} at Bar {idx} (t={ts_ms/1000:.0f}s), Price ${entry_p:.2f}")
            
        if pos is not None:
            elapsed = (ts_ms - pos.entry_ts_ms) / 1000.0
            sign = 1.0 if pos.side == "BUY" else -1.0
            unreal = sign * ((mid - pos.entry_price) / pos.entry_price) * 1e4
            # print every ~100s or at key points
            if int(elapsed) % 150 < 10 or elapsed >= 890 and elapsed <= 910:
                d_tact = tact[2] - tact[0]
                d_meso = meso[2] - meso[0]
                print(f"  t+{elapsed:.0f}s (Bar {idx}): Mid=${mid:.2f} | Unreal={unreal:+.2f} bps | MFE={pos.mfe:+.2f} | MAE={pos.mae:+.2f} | d_tact={d_tact:+.2f} | d_meso={d_meso:+.2f} | p_dir={p_dir}")
                
        if prev_pos is not None and pos is None:
            t = engine.closed_trades[-1]
            print(f"EXITED {t.side} at Bar {idx} (t={ts_ms/1000:.0f}s): Net={t.net_return_bps:+.2f} bps | Reason={t.exit_reason}")
