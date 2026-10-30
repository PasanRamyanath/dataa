import os
import glob
import json
import sys
import types
from collections import Counter

sys.path.append('feature engineering scripts')
from test_quantile_envelope_tuning import QuantileTunedEngine

base_dir = 'scratch'
archives = sorted([d for d in os.listdir(base_dir) if (d.startswith('archive_cycle') or d.startswith('archive_v3')) and os.path.isdir(os.path.join(base_dir, d))])

class OptimizedPeakBidirectionalEngine(QuantileTunedEngine):
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        min_net_edge_bps: float = 7.0,
        trend_threshold_bps: float = 12.0,
        short_min_pct: float = 0.32,
        long_max_pct: float = 0.75,
        min_dir_bars: int = 2,
        cooldown_sec: float = 45.0,
        max_entry_tick: int = 335,
        max_holding_sec: float = 900.0,
        hard_stop_bps: float = 45.0
    ):
        super().__init__(
            short_min_pct=short_min_pct,
            long_max_pct=long_max_pct,
            min_dir_bars=min_dir_bars,
            trend_threshold_bps=trend_threshold_bps,
            min_net_edge_bps=min_net_edge_bps
        )
        self.strategy_profile = "OptimizedPeakBidirectionalEngine"
        self.cooldown_sec = cooldown_sec
        self.max_entry_tick = max_entry_tick
        self.max_holding_sec = max_holding_sec
        self.hard_stop_bps = hard_stop_bps

def evaluate_params(edge, trend_th, short_pct, cooldown):
    all_closed_trades = []
    long_trades = []
    short_trades = []
    active_sessions = 0
    wr100_sessions = 0
    profitable_sessions = 0

    for arch in archives:
        ledgers = glob.glob(os.path.join(base_dir, arch, '*ledger*.jsonl'))
        if not ledgers: continue
        with open(ledgers[0], 'r', encoding='utf-8') as f:
            bars = [json.loads(l) for l in f if l.strip().startswith('{')]

        engine = OptimizedPeakBidirectionalEngine(
            min_net_edge_bps=edge,
            trend_threshold_bps=trend_th,
            short_min_pct=short_pct,
            long_max_pct=0.75,
            min_dir_bars=2,
            cooldown_sec=cooldown
        )

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
            e = bar.get('expected_net_edge_bps', bar.get('edge', 0.0))

            decision = types.SimpleNamespace(
                ts_ms=ts_ms, primary_direction=p_dir, expected_net_edge_bps=e,
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
                item = {'cycle': arch, 'side': t.side, 'net': t.net_return_bps, 'reason': t.exit_reason}
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
    loss_strs = [f"{x['cycle']}_{x['side']}" for x in losses_list]

    print(f"Edge={edge:.1f}b, TrendTh={trend_th:.1f}b, ShortMin={short_pct:.2f}, CD={cooldown:.0f}s:")
    print(f"  -> Total: {n} trades across {active_sessions} sessions ({n/max(1,active_sessions):.2f}/sess) | WR={wr:.2f}% ({w}W/{l}L) | Net={tot_pnl:+.2f} bps | Avg={avg_pnl:+.2f}b | PF={pf:.2f}")
    print(f"     LONGS : {n_l} trades | WR={wr_l:.1f}% ({w_l}W/{n_l-w_l}L) | Net={pnl_l:+.2f} bps")
    print(f"     SHORTS: {n_s} trades | WR={wr_s:.1f}% ({w_s}W/{n_s-w_s}L) | Net={pnl_s:+.2f} bps")
    print(f"     100% WR Sessions: {wr100_sessions}/{active_sessions} ({wr100_sessions/max(1,active_sessions)*100:.1f}%) | Losses: {loss_strs}")
    print()

if __name__ == "__main__":
    for e in [7.5, 7.0, 6.5, 6.0]:
        for tt in [10.0, 12.0, 14.0]:
            for sp in [0.30, 0.33, 0.35]:
                for cd in [45.0, 30.0]:
                    evaluate_params(edge=e, trend_th=tt, short_pct=sp, cooldown=cd)
