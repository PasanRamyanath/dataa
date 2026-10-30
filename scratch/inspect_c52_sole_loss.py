import os
import glob
import json
import sys
import types

sys.path.append('feature engineering scripts')
from test_quantile_envelope_tuning import QuantileTunedEngine

base_dir = 'scratch'
path = os.path.join(base_dir, 'archive_cycle52')
ledgers = glob.glob(os.path.join(path, '*ledger*.jsonl'))
with open(ledgers[0], 'r', encoding='utf-8') as f:
    bars = [json.loads(l) for l in f if l.strip().startswith('{')]

engine = QuantileTunedEngine(short_min_pct=0.35, long_max_pct=0.75, min_dir_bars=2)

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
        print(f"Trade in Cycle 52 Closed: {t.side} | Net={t.net_return_bps:+.2f} bps | Gross={t.gross_return_bps:+.2f} | Reason={t.exit_reason}")
        print(f"Entry: ${t.entry_price:.2f} -> Exit: ${t.exit_price:.2f} | Dur={t.holding_duration_sec:.1f}s")
        # Entry bar details
        ent_idx = max(0, idx - int(t.holding_duration_sec/10))
        b_ent = bars[ent_idx]
        print(f"Entry Bar: {ent_idx}")
        print(f"  Macro: {b_ent.get('macro_probs')}")
        print(f"  Meso:  {b_ent.get('meso_probs')}")
        print(f"  Tact:  {b_ent.get('tactical_probs')}")
        print(f"  Micro: {b_ent.get('micro_probs')}")
