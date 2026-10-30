import os
import glob
import json
import sys
import types
from collections import Counter

sys.path.append('feature engineering scripts')
from execution.champion_challenger_framework import Cycle35_ReversionPureScalper

base_dir = 'scratch'
archives = sorted([d for d in os.listdir(base_dir) if (d.startswith('archive_cycle') or d.startswith('archive_v3')) and os.path.isdir(os.path.join(base_dir, d))])

all_closed_trades = []
long_trades = []
short_trades = []

for arch in archives:
    ledgers = glob.glob(os.path.join(base_dir, arch, '*ledger*.jsonl'))
    if not ledgers: continue
    with open(ledgers[0], 'r', encoding='utf-8') as f:
        bars = [json.loads(l) for l in f if l.strip().startswith('{')]

    engine = Cycle35_ReversionPureScalper(symbol='BTCUSDT')

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
        all_closed_trades.append(t)
        if t.side == 'BUY': long_trades.append(t)
        else: short_trades.append(t)

def stats(trades):
    if not trades: return '0 trades'
    wins = [t for t in trades if t.net_return_bps > 0]
    losses = [t for t in trades if t.net_return_bps <= 0]
    wr = len(wins)/len(trades)*100
    pnl = sum(t.net_return_bps for t in trades)
    avg = pnl/len(trades)
    gw = sum(t.net_return_bps for t in wins)
    gl = abs(sum(t.net_return_bps for t in losses))
    pf = (gw / gl) if gl > 0 else (999.0 if gw > 0 else 1.0)
    return f'{len(trades)} trades | WR: {wr:.1f}% ({len(wins)}W/{len(losses)}L) | Net: {pnl:+.2f} bps (avg {avg:+.2f}) | PF: {pf:.2f}'

print("=== Cycle35_ReversionPureScalper Baseline Audit across 64 Cycles ===")
print(f"  TOTAL: {stats(all_closed_trades)}")
print(f"  LONGS: {stats(long_trades)}")
print(f"  SHORTS: {stats(short_trades)}")
losses = [t for t in all_closed_trades if t.net_return_bps <= 0]
print(f"\nLosses count: {len(losses)}")
for l in losses:
    print(f"  {l.side} | Net={l.net_return_bps:+.2f} bps | Gross={l.gross_return_bps:+.2f} | Reason={l.exit_reason} | Dur={l.holding_duration_sec:.1f}s")
