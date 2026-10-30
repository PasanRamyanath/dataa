import os
import glob
import json
import sys
import types

sys.path.append('feature engineering scripts')
from execution.champion_challenger_framework import Cycle65_UnifiedPeakReversionEngine

base_dir = 'scratch'
archives = sorted([d for d in os.listdir(base_dir) if (d.startswith('archive_cycle') or d.startswith('archive_v3')) and os.path.isdir(os.path.join(base_dir, d))])

long_trades = []
short_trades = []
losses = []

for arch in archives:
    ledgers = glob.glob(os.path.join(base_dir, arch, '*ledger*.jsonl'))
    if not ledgers: 
        continue
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
        dec = types.SimpleNamespace(
            ts_ms=ts_ms, primary_direction=p_dir, expected_net_edge_bps=edge,
            tactical_probs=tact, meso_probs=meso, macro_probs=macro, micro_probs=micro,
            kelly_position_fraction=0.15, execution_mode='PASSIVE_MAKER_SNIPER',
            target_holding_seconds=900, take_profit_bps=35.0, stop_loss_bps=45.0,
            micro_sniper_trigger_prob=0.85
        )
        engine.on_market_micro_tick(ts_ms, bid, ask, bid_qty, ask_qty, ask + 0.05, bid - 0.05, dec)
        engine.on_hierarchical_decision(dec, bid, ask, bid_qty, ask_qty, 10000.0, mid)
        if engine.state == 'RESTING_ENTRY_QUEUE':
            engine.on_market_micro_tick(ts_ms + 1000, bid, ask, bid_qty, ask_qty, ask + 0.05, bid - 0.05, dec)

    for t in engine.closed_trades:
        item = {
            'cycle': arch, 'side': t.side, 'net': t.net_return_bps, 'gross': t.gross_return_bps,
            'entry_p': t.entry_price, 'exit_p': t.exit_price, 'mfe': getattr(t, 'mfe', 0.0),
            'mae': getattr(t, 'mae', 0.0), 'reason': t.exit_reason, 'dur': t.holding_duration_sec
        }
        if t.side == 'BUY': 
            long_trades.append(item)
        else: 
            short_trades.append(item)
        if t.net_return_bps <= 0: 
            losses.append(item)

print(f'Total Long Trades: {len(long_trades)}')
w_l = sum(1 for t in long_trades if t['net'] > 0)
net_l = sum(t['net'] for t in long_trades)
print(f'  Longs: {w_l}W / {len(long_trades)-w_l}L | WR: {w_l/len(long_trades)*100:.1f}% | Net: {net_l:+.2f} bps')

print(f'Total Short Trades: {len(short_trades)}')
w_s = sum(1 for t in short_trades if t['net'] > 0)
net_s = sum(t['net'] for t in short_trades)
print(f'  Shorts: {w_s}W / {len(short_trades)-w_s}L | WR: {w_s/len(short_trades)*100:.1f}% | Net: {net_s:+.2f} bps')

print('\nDetailed Breakdown of ALL Losses in Challenger 1:')
for i, l in enumerate(losses):
    print(f'Loss {i+1}: {l["cycle"]} | Side: {l["side"]} | Net: {l["net"]:+.2f} bps | Gross: {l["gross"]:+.2f} | Reason: {l["reason"]} | MAE: {l["mae"]:.2f} | MFE: {l["mfe"]:.2f}')
