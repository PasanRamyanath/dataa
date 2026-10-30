import os
import glob
import json
import sys
import types
from collections import Counter

sys.path.append('feature engineering scripts')
from execution.champion_challenger_framework import Cycle65_UnifiedPeakReversionEngine

base_dir = 'scratch'
archives = sorted([d for d in os.listdir(base_dir) if (d.startswith('archive_cycle') or d.startswith('archive_v3')) and os.path.isdir(os.path.join(base_dir, d))])

def test_macro_alignment_filter(macro_tol_list):
    for macro_tol in macro_tol_list:
        all_closed_trades = []
        long_trades = []
        short_trades = []
        losses = []

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
                    ts_ms=ts_ms,
                    primary_direction=p_dir,
                    expected_net_edge_bps=edge,
                    tactical_probs=tact,
                    meso_probs=meso,
                    macro_probs=macro,
                    micro_probs=micro,
                    kelly_position_fraction=0.15,
                    execution_mode='PASSIVE_MAKER_SNIPER',
                    target_holding_seconds=900,
                    take_profit_bps=35.0,
                    stop_loss_bps=45.0,
                    micro_sniper_trigger_prob=0.85
                )

                # Macro filter modification in on_hierarchical_decision
                d_tact = tact[2] - tact[0]
                d_meso = meso[2] - meso[0]
                d_macro = macro[2] - macro[0]

                # Filter entry if macro contradicts
                if macro_tol is not None and engine.state == "IDLE":
                    if p_dir == 2 and d_macro < -macro_tol:
                        # Skip long
                        decision.primary_direction = 1
                    elif p_dir == 0 and d_macro > macro_tol:
                        # Skip short
                        decision.primary_direction = 1

                engine.on_market_micro_tick(ts_ms=ts_ms, bbo_bid=bid, bbo_ask=ask, bbo_bid_qty=bid_qty, bbo_ask_qty=ask_qty, bar_high=ask+0.05, bar_low=bid-0.05, current_decision=decision)
                engine.on_hierarchical_decision(decision, bid, ask, bid_qty, ask_qty, 10000.0, mid)
                if engine.state == 'RESTING_ENTRY_QUEUE':
                    engine.on_market_micro_tick(ts_ms=ts_ms+1000, bbo_bid=bid, bbo_ask=ask, bbo_bid_qty=bid_qty, bbo_ask_qty=ask_qty, bar_high=ask+0.05, bar_low=bid-0.05, current_decision=decision)

            for t in engine.closed_trades:
                item = {
                    'cycle': arch, 'side': t.side, 'gross': t.gross_return_bps,
                    'net': t.net_return_bps, 'fee': t.total_fee_bps, 'dur': t.holding_duration_sec,
                    'reason': t.exit_reason
                }
                all_closed_trades.append(item)
                if t.side == 'BUY': long_trades.append(item)
                else: short_trades.append(item)
                if t.net_return_bps <= 0: losses.append(item)

        n = len(all_closed_trades)
        w = sum(1 for t in all_closed_trades if t['net'] > 0)
        wr = (w / n * 100.0) if n > 0 else 0.0
        tot_net = sum(t['net'] for t in all_closed_trades)
        avg_net = (tot_net / n) if n > 0 else 0.0

        n_l = len(long_trades)
        w_l = sum(1 for t in long_trades if t['net'] > 0)
        wr_l = (w_l / n_l * 100.0) if n_l > 0 else 0.0
        net_l = sum(t['net'] for t in long_trades)

        n_s = len(short_trades)
        w_s = sum(1 for t in short_trades if t['net'] > 0)
        wr_s = (w_s / n_s * 100.0) if n_s > 0 else 0.0
        net_s = sum(t['net'] for t in short_trades)

        gw = sum(t['net'] for t in all_closed_trades if t['net'] > 0)
        gl = abs(sum(t['net'] for t in all_closed_trades if t['net'] <= 0))
        pf = (gw / gl) if gl > 0 else 999.0

        lbl = f"Macro Tolerance = {macro_tol}" if macro_tol is not None else "Baseline"
        print(f"=== {lbl} ===")
        print(f"  TOTAL: {n} trades | Wins={w} Losses={n-w} | WR={wr:.2f}% | Net={tot_net:+.2f} bps | Avg={avg_net:+.2f} bps | PF={pf:.2f}")
        print(f"  LONGS: {n_l} trades | Wins={w_l} Losses={n_l-w_l} | WR={wr_l:.1f}% | Net={net_l:+.2f} bps")
        print(f"  SHORTS:{n_s} trades | Wins={w_s} Losses={n_s-w_s} | WR={wr_s:.1f}% | Net={net_s:+.2f} bps")
        print(f"  Losses count: {len(losses)}: {[l['cycle'] + '_' + l['side'] for l in losses]}")
        print()

test_macro_alignment_filter([None, 0.10, 0.05, 0.0, -0.05])
