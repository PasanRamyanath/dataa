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

class BaselineTest(Cycle65_UnifiedPeakReversionEngine):
    def __init__(self):
        super().__init__(symbol="BTCUSDT")
        self.session_open_price = None

    def _is_range_boundary_violation(self, primary_direction: int, mid: float) -> bool:
        if len(self.rolling_15m_prices) >= 10:
            sess_high = max(self.rolling_15m_prices)
            sess_low = min(self.rolling_15m_prices)
            rng = sess_high - sess_low
            if rng >= 15.0:
                pct = (mid - sess_low) / rng
                if primary_direction == 0 and pct <= 0.25:
                    return True
                if primary_direction == 2 and pct >= 0.75:
                    return True
        return False

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        now_ms = decision.ts_ms
        self.ticks_seen += 1
        mid = 0.5 * (bbo_bid + bbo_ask)

        if self.session_open_price is None:
            self.session_open_price = mid

        if self._last_rolling_ts != now_ms:
            self._last_rolling_ts = now_ms
            self.rolling_15m_prices.append(mid)
            self.recent_mid_prices.append((now_ms, mid))

        d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
        d_meso = decision.meso_probs[2] - decision.meso_probs[0]
        drift_from_open_bps = ((mid - self.session_open_price) / self.session_open_price) * 1e4

        if self.state == "IDLE":
            if self.ticks_seen < 10:
                return self.state
            if self.ticks_seen > 335:
                return self.state

            allowed_dirs = []
            if drift_from_open_bps > 10.0:
                allowed_dirs = [2]
            elif drift_from_open_bps < -10.0:
                allowed_dirs = [0]
            else:
                allowed_dirs = [0, 2]

            if decision.primary_direction in allowed_dirs and decision.expected_net_edge_bps >= 8.0:
                cand_dir = None
                if decision.primary_direction == 2 and d_tact >= 0.04 and d_meso >= 0.01:
                    cand_dir = 2
                elif decision.primary_direction == 0 and d_tact <= -0.04 and d_meso <= -0.01:
                    cand_dir = 0

                if cand_dir in allowed_dirs:
                    if self._is_range_boundary_violation(cand_dir, mid):
                        return self.state
                    if self._is_parabolic_surge(cand_dir, now_ms, mid):
                        return self.state

                    if self.last_exit_ts_ms > 0:
                        elapsed = (now_ms - self.last_exit_ts_ms) / 1000.0
                        if elapsed < 60.0 and self.last_trade_exit_price > 0:
                            if cand_dir == 2 and self.last_trade_side == "BUY" and (mid > self.last_trade_exit_price - self.range_boundary_buffer_dollars):
                                return self.state
                            elif cand_dir == 0 and self.last_trade_side == "SELL" and (mid < self.last_trade_exit_price + self.range_boundary_buffer_dollars):
                                return self.state

                    import copy
                    tact_dec = copy.copy(decision)
                    tact_dec.primary_direction = cand_dir
                    self.stalking_decision = tact_dec
                    self._place_sniper_maker_entry(tact_dec, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, now_ms, intra_bar_open)
                    return self.state

        if self.state in ("FILLED_SWING_HOLDING", "PASSIVE_MAKER_UNWIND"):
            self._manage_active_swing_position(decision, bbo_bid, bbo_ask, now_ms)
        return self.state

target_cycles = ['archive_cycle17', 'archive_cycle19', 'archive_cycle43', 'archive_cycle52', 'archive_cycle57']

for arch in target_cycles:
    ledgers = glob.glob(os.path.join(base_dir, arch, '*ledger*.jsonl'))
    if not ledgers: continue
    with open(ledgers[0], 'r', encoding='utf-8') as f:
        bars = [json.loads(l) for l in f if l.strip().startswith('{')]

    engine = BaselineTest()

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
            if t.net_return_bps <= 0:
                print(f"=== {arch} Short Loss Trade Audit ===")
                print(f"Entry: ${t.entry_price:.2f} -> Exit: ${t.exit_price:.2f} | Net: {t.net_return_bps:+.2f} bps | Dur: {t.holding_duration_sec:.1f}s")
                # find entry bar index
                ent_idx = max(0, idx - int(t.holding_duration_sec/10))
                b_ent = bars[ent_idx]
                m_open = bars[0].get('bbo_mid', bars[0].get('mid'))
                m_ent = b_ent.get('bbo_mid', b_ent.get('mid'))
                print(f"Entry Bar: {ent_idx} | Price: ${m_ent:.2f} | Session Open: ${m_open:.2f} | Drift: {((m_ent-m_open)/m_open)*1e4:+.2f} bps")
                print(f"MFE: {getattr(t, 'peak_mfe_bps', 0.0)} bps | MAE: {getattr(t, 'max_mae_bps', 0.0)} bps")
                print(f"Model Probs at Entry:")
                print(f"  Macro: {b_ent.get('macro_probs')} (d={b_ent.get('macro_probs')[2]-b_ent.get('macro_probs')[0]:+.3f})")
                print(f"  Meso:  {b_ent.get('meso_probs')} (d={b_ent.get('meso_probs')[2]-b_ent.get('meso_probs')[0]:+.3f})")
                print(f"  Tact:  {b_ent.get('tactical_probs')} (d={b_ent.get('tactical_probs')[2]-b_ent.get('tactical_probs')[0]:+.3f})")
                print(f"  Micro: {b_ent.get('micro_probs')} (d={b_ent.get('micro_probs')[2]-b_ent.get('micro_probs')[0]:+.3f})")
                # Look at 5 bars before entry:
                ret_5_before = ((m_ent - bars[max(0, ent_idx-5)].get('bbo_mid', bars[max(0, ent_idx-5)].get('mid'))) / m_ent) * 1e4
                ret_10_before = ((m_ent - bars[max(0, ent_idx-10)].get('bbo_mid', bars[max(0, ent_idx-10)].get('mid'))) / m_ent) * 1e4
                print(f"Momentum before entry: 5-bar Ret={ret_5_before:+.2f} bps, 10-bar Ret={ret_10_before:+.2f} bps")
                print()
