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
    entry_history = {}
    
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
        prev_closed_len = len(engine.closed_trades)
        
        engine.on_market_micro_tick(ts_ms, bid, ask, bid_qty, ask_qty, ask + 0.05, bid - 0.05, dec)
        engine.on_hierarchical_decision(dec, bid, ask, bid_qty, ask_qty, 10000.0, mid)
        if engine.state == 'RESTING_ENTRY_QUEUE':
            engine.on_market_micro_tick(ts_ms + 1000, bid, ask, bid_qty, ask_qty, ask + 0.05, bid - 0.05, dec)
            
        if engine.active_position is not None and prev_pos is None:
            # Just filled!
            entry_history[engine.active_position.entry_ts_ms] = {
                'bar_idx': idx,
                'mid': mid,
                'p_dir': p_dir,
                'edge': edge,
                'tact': tact,
                'meso': meso,
                'macro': macro,
                'micro': micro
            }
            
        if len(engine.closed_trades) > prev_closed_len:
            t = engine.closed_trades[-1]
            if t.net_return_bps <= 0:
                ent = entry_history.get(t.entry_ts_ms, {})
                print(f"=== {cycle_name} LOSS TRADE AUDIT ===")
                print(f"Side: {t.side} | Entry Price: ${t.entry_price:.2f} | Exit Price: ${t.exit_price:.2f}")
                print(f"Net Return: {t.net_return_bps:+.2f} bps | Gross: {t.gross_return_bps:+.2f} bps")
                print(f"Exit Reason: {t.exit_reason} | Duration: {t.holding_duration_sec:.1f}s")
                print(f"Entry at Bar Index: {ent.get('bar_idx')} | Mid: {ent.get('mid')}")
                print(f"Entry Signals at Placement/Fill:")
                print(f"  Primary Dir: {ent.get('p_dir')} | Net Edge: {ent.get('edge', 0.0):.2f} bps")
                print(f"  Tactical: {ent.get('tact')} (d_tact={ent.get('tact', [0,0,0])[2]-ent.get('tact', [0,0,0])[0]:+.4f})")
                print(f"  Meso:     {ent.get('meso')} (d_meso={ent.get('meso', [0,0,0])[2]-ent.get('meso', [0,0,0])[0]:+.4f})")
                print(f"  Macro:    {ent.get('macro')} (d_macro={ent.get('macro', [0,0,0])[2]-ent.get('macro', [0,0,0])[0]:+.4f})")
                print(f"  Micro:    {ent.get('micro')} (d_micro={ent.get('micro', [0,0,0])[2]-ent.get('micro', [0,0,0])[0]:+.4f})")
                print()
