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

def test_meso_invalidation_sweep(inval_threshold_list):
    for inval_thresh in inval_threshold_list:
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

                # Custom Invalidation Check before manage_active
                if inval_thresh is not None and engine.active_position is not None:
                    pos = engine.active_position
                    elapsed = (ts_ms - pos.entry_ts_ms) / 1000.0
                    if elapsed >= 60.0: # give at least 1 minute breathing room
                        d_meso = meso[2] - meso[0]
                        if pos.side == "BUY" and d_meso <= -inval_thresh:
                            # Invalidation cut
                            sign = 1.0
                            unreal = sign * ((mid - pos.entry_price) / pos.entry_price) * 1e4
                            exit_p = bid if unreal < -8.0 else ask
                            fee = 4.5 if unreal < -8.0 else 2.0
                            engine._close_position(exit_price=exit_p, exit_ts_ms=ts_ms, exit_fee_bps=fee, reason="MESO_REGIME_INVALIDATION")
                        elif pos.side == "SELL" and d_meso >= inval_thresh:
                            sign = -1.0
                            unreal = sign * ((mid - pos.entry_price) / pos.entry_price) * 1e4
                            exit_p = ask if unreal < -8.0 else bid
                            fee = 4.5 if unreal < -8.0 else 2.0
                            engine._close_position(exit_price=exit_p, exit_ts_ms=ts_ms, exit_fee_bps=fee, reason="MESO_REGIME_INVALIDATION")

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

        lbl = f"Threshold = {inval_thresh}" if inval_thresh is not None else "Baseline (No Invalidation)"
        print(f"=== {lbl} ===")
        print(f"  TOTAL: {n} trades | Wins={w} Losses={n-w} | WR={wr:.2f}% | Net={tot_net:+.2f} bps | Avg={avg_net:+.2f} bps | PF={pf:.2f}")
        print(f"  LONGS: {n_l} trades | Wins={w_l} Losses={n_l-w_l} | WR={wr_l:.1f}% | Net={net_l:+.2f} bps")
        print(f"  SHORTS:{n_s} trades | Wins={w_s} Losses={n_s-w_s} | WR={wr_s:.1f}% | Net={net_s:+.2f} bps")
        reasons = Counter(t['reason'] for t in all_closed_trades)
        print(f"  Exit counts: {dict(reasons)}")
        print()

test_meso_invalidation_sweep([None, 0.25, 0.20, 0.18, 0.15])
