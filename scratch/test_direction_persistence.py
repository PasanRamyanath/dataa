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

class PersistenceEngine(Cycle65_UnifiedPeakReversionEngine):
    def __init__(self, min_dir_bars=2, trend_threshold_bps=10.0, min_net_edge_bps=8.0):
        super().__init__(symbol="BTCUSDT")
        self.min_dir_bars = min_dir_bars
        self.trend_threshold_bps = trend_threshold_bps
        self.min_net_edge_bps = min_net_edge_bps
        self.session_open_price = None
        self.consecutive_dir = 0
        self.current_dir = 1

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

        # Track direction persistence
        if decision.primary_direction == self.current_dir:
            self.consecutive_dir += 1
        else:
            self.current_dir = decision.primary_direction
            self.consecutive_dir = 1

        d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
        d_meso = decision.meso_probs[2] - decision.meso_probs[0]
        drift_from_open_bps = ((mid - self.session_open_price) / self.session_open_price) * 1e4

        if self.state == "IDLE":
            if self.ticks_seen < 10:
                return self.state
            if self.ticks_seen > 335:
                return self.state

            # Persistence gate: must be at least min_dir_bars
            if self.consecutive_dir < self.min_dir_bars:
                return self.state

            allowed_dirs = []
            if drift_from_open_bps > self.trend_threshold_bps:
                allowed_dirs = [2]
            elif drift_from_open_bps < -self.trend_threshold_bps:
                allowed_dirs = [0]
            else:
                allowed_dirs = [0, 2]

            if decision.primary_direction in allowed_dirs and decision.expected_net_edge_bps >= self.min_net_edge_bps:
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

def test_persistence_sweep():
    for mdb in [1, 2, 3]:
        all_closed_trades = []
        long_trades = []
        short_trades = []
        active_sessions = 0
        wr100_sessions = 0
        profitable_sessions = 0
        exit_reasons = Counter()

        for arch in archives:
            ledgers = glob.glob(os.path.join(base_dir, arch, '*ledger*.jsonl'))
            if not ledgers: continue
            with open(ledgers[0], 'r', encoding='utf-8') as f:
                bars = [json.loads(l) for l in f if l.strip().startswith('{')]

            engine = PersistenceEngine(min_dir_bars=mdb, trend_threshold_bps=10.0, min_net_edge_bps=8.0)

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

            session_trades = engine.closed_trades
            if session_trades:
                active_sessions += 1
                w = sum(1 for t in session_trades if t.net_return_bps > 0)
                sess_net = sum(t.net_return_bps for t in session_trades)
                if w == len(session_trades): wr100_sessions += 1
                if sess_net > 0: profitable_sessions += 1

                for t in session_trades:
                    exit_reasons[t.exit_reason] += 1
                    item = {
                        'cycle': arch, 'side': t.side, 'gross': t.gross_return_bps,
                        'net': t.net_return_bps, 'fee': t.total_fee_bps, 'dur': t.holding_duration_sec,
                        'reason': t.exit_reason
                    }
                    all_closed_trades.append(item)
                    if t.side == 'BUY': long_trades.append(item)
                    else: short_trades.append(item)

        n = len(all_closed_trades)
        w = sum(1 for t in all_closed_trades if t['net'] > 0)
        l = n - w
        wr = (w / n * 100.0) if n > 0 else 0.0
        tot_pnl = sum(t['net'] for t in all_closed_trades)
        avg_pnl = (tot_pnl / n) if n > 0 else 0.0

        n_l = len(long_trades)
        w_l = sum(1 for t in long_trades if t['net'] > 0)
        wr_l = (w_l / n_l * 100.0) if n_l > 0 else 0.0
        pnl_l = sum(t['net'] for t in long_trades)

        n_s = len(short_trades)
        w_s = sum(1 for t in short_trades if t['net'] > 0)
        wr_s = (w_s / n_s * 100.0) if n_s > 0 else 0.0
        pnl_s = sum(t['net'] for t in short_trades)

        gw = sum(t['net'] for t in all_closed_trades if t['net'] > 0)
        gl = abs(sum(t['net'] for t in all_closed_trades if t['net'] <= 0))
        pf = (gw / gl) if gl > 0 else 999.0

        losses_list = [t for t in all_closed_trades if t['net'] <= 0]
        loss_strs = [f"{x['cycle']}_{x['side']}({x['net']:+.1f}b)" for x in losses_list]

        print("=" * 95)
        print(f"PERSISTENCE SWEEP: min_dir_bars = {mdb}")
        print(f"Overall: {n} trades across {active_sessions} sessions | Wins={w} Losses={l} | WR={wr:.2f}% | Net={tot_pnl:+.2f} bps | Avg={avg_pnl:+.2f}b | PF={pf:.2f}")
        print(f"  LONGS : {n_l} trades | Wins={w_l} Losses={n_l-w_l} | WR={wr_l:.1f}% | Net={pnl_l:+.2f} bps")
        print(f"  SHORTS: {n_s} trades | Wins={w_s} Losses={n_s-w_s} | WR={wr_s:.1f}% | Net={pnl_s:+.2f} bps")
        print(f"  100% WR Sessions: {wr100_sessions}/{active_sessions} ({wr100_sessions/max(1,active_sessions)*100:.1f}%) | Profitable: {profitable_sessions}/{active_sessions} ({profitable_sessions/max(1,active_sessions)*100:.1f}%)")
        print(f"  Losses: {loss_strs}")

if __name__ == "__main__":
    test_persistence_sweep()
