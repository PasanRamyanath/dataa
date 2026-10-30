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

class Cycle65_PeakBidirectionalEngine(Cycle65_UnifiedPeakReversionEngine):
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        min_net_edge_bps: float = 6.0,
        max_entry_tick: int = 330,
        cooldown_sec: float = 60.0,
        min_range_bars: int = 12,
        macro_filter_tolerance: float = 0.0,
        session_drift_limit_bps: float = 25.0,
        stale_breathing_cutoff_sec: float = 1200.0,
        hard_stop_bps: float = 45.0
    ):
        super().__init__(symbol=symbol)
        self.strategy_profile = "Cycle65_PeakBidirectionalEngine"
        self.min_net_edge_bps = min_net_edge_bps
        self.max_entry_tick = max_entry_tick
        self.cooldown_sec = cooldown_sec
        self.min_range_bars = min_range_bars
        self.macro_filter_tolerance = macro_filter_tolerance
        self.session_drift_limit_bps = session_drift_limit_bps
        self.stale_breathing_cutoff_sec = stale_breathing_cutoff_sec
        self.hard_stop_bps = hard_stop_bps
        self.session_open_price = None

    def _is_range_boundary_violation(self, primary_direction: int, mid: float) -> bool:
        if len(self.rolling_15m_prices) >= self.min_range_bars:
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
        d_macro = decision.macro_probs[2] - decision.macro_probs[0]
        sess_drift_bps = ((mid - self.session_open_price) / self.session_open_price) * 1e4

        if self.state == "IDLE":
            if self.ticks_seen < self.min_range_bars:
                return self.state
            if self.ticks_seen > self.max_entry_tick:
                return self.state

            # Bidirectional Entry Evaluation
            if decision.primary_direction in (0, 2) and decision.expected_net_edge_bps >= self.min_net_edge_bps:
                cand_dir = None
                if decision.primary_direction == 2 and d_tact >= 0.04 and d_meso >= 0.01:
                    # Guard against buying into strong bear session drift or hostile macro
                    if d_macro >= -self.macro_filter_tolerance and sess_drift_bps >= -self.session_drift_limit_bps:
                        cand_dir = 2
                elif decision.primary_direction == 0 and d_tact <= -0.04 and d_meso <= -0.01:
                    # Guard against shorting into strong bull session drift or hostile macro
                    if d_macro <= self.macro_filter_tolerance and sess_drift_bps <= self.session_drift_limit_bps:
                        cand_dir = 0

                if cand_dir in (0, 2):
                    if self._is_range_boundary_violation(cand_dir, mid):
                        return self.state
                    if self._is_parabolic_surge(cand_dir, now_ms, mid):
                        return self.state

                    if self.last_exit_ts_ms > 0:
                        elapsed = (now_ms - self.last_exit_ts_ms) / 1000.0
                        if elapsed < self.cooldown_sec and self.last_trade_exit_price > 0:
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

def run_test(params, label):
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

        engine = Cycle65_PeakBidirectionalEngine(**params)

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

            engine.on_market_micro_tick(ts_ms=ts_ms, bbo_bid=bid, bbo_ask=ask, bbo_bid_qty=bid_qty, bbo_ask_qty=ask_qty, bar_high=ask+0.05, bar_low=bid-0.05, current_decision=decision)
            engine.on_hierarchical_decision(decision, bid, ask, bid_qty, ask_qty, 10000.0, mid)
            if engine.state == 'RESTING_ENTRY_QUEUE':
                engine.on_market_micro_tick(ts_ms=ts_ms+1000, bbo_bid=bid, bbo_ask=ask, bbo_bid_qty=bid_qty, bbo_ask_qty=ask_qty, bar_high=ask+0.05, bar_low=bid-0.05, current_decision=decision)

        session_trades = engine.closed_trades
        if session_trades:
            active_sessions += 1
            w = sum(1 for t in session_trades if t.net_return_bps > 0)
            sess_net = sum(t.net_return_bps for t in session_trades)
            if w == len(session_trades):
                wr100_sessions += 1
            if sess_net > 0:
                profitable_sessions += 1

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

    n_long = len(long_trades)
    w_long = sum(1 for t in long_trades if t['net'] > 0)
    wr_long = (w_long / n_long * 100.0) if n_long > 0 else 0.0
    pnl_long = sum(t['net'] for t in long_trades)

    n_short = len(short_trades)
    w_short = sum(1 for t in short_trades if t['net'] > 0)
    wr_short = (w_short / n_short * 100.0) if n_short > 0 else 0.0
    pnl_short = sum(t['net'] for t in short_trades)

    gw = sum(t['net'] for t in all_closed_trades if t['net'] > 0)
    gl = abs(sum(t['net'] for t in all_closed_trades if t['net'] <= 0))
    pf = (gw / gl) if gl > 0 else 999.0

    print("=" * 100)
    print(f"BENCHMARK: {label}")
    print("=" * 100)
    print(f"Total Closed Trades: {n} across {active_sessions} active sessions ({n/max(1, active_sessions):.2f} trades/session)")
    print(f"Overall: Wins={w} | Losses={l} | Win Rate={wr:.2f}% | Net Realized PnL={tot_pnl:+.2f} bps | Avg={avg_pnl:+.2f} bps/trade | PF={pf:.2f}")
    print(f"  LONGS : {n_long} trades | Wins={w_long} | WR={wr_long:.1f}% | Net PnL={pnl_long:+.2f} bps (avg {pnl_long/max(1,n_long):+.2f}b)")
    print(f"  SHORTS: {n_short} trades | Wins={w_short} | WR={wr_short:.1f}% | Net PnL={pnl_short:+.2f} bps (avg {pnl_short/max(1,n_short):+.2f}b)")
    print(f"100% WR Sessions: {wr100_sessions}/{active_sessions} ({wr100_sessions/max(1,active_sessions)*100:.1f}%) | Profitable Sessions: {profitable_sessions}/{active_sessions} ({profitable_sessions/max(1,active_sessions)*100:.1f}%)")
    losses_list = [t for t in all_closed_trades if t['net'] <= 0]
    loss_strs = [f"{x['cycle']}_{x['side']}({x['net']:+.1f}b)" for x in losses_list]
    print(f"Losses ({len(losses_list)}): {loss_strs}")
    print("Exit Reasons:")
    for r, cnt in exit_reasons.most_common():
        sub_pnl = sum(t['net'] for t in all_closed_trades if t['reason'] == r)
        print(f"  {r:<35}: {cnt:>3} trades | Net={sub_pnl:>+8.2f} bps | Avg={sub_pnl/cnt:>+5.2f} bps")
    print("=" * 100)

if __name__ == "__main__":
    for edge in [7.0, 6.0, 5.0]:
        for drift_lim in [30.0, 25.0, 20.0]:
            run_test({
                'min_net_edge_bps': edge,
                'max_entry_tick': 330,
                'cooldown_sec': 60.0,
                'min_range_bars': 12,
                'macro_filter_tolerance': 0.02,
                'session_drift_limit_bps': drift_lim,
                'hard_stop_bps': 45.0
            }, f"Bidirectional Engine (Edge={edge}b, DriftLimit={drift_lim}b)")
