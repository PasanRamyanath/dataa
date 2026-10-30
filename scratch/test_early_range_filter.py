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

class UnifiedEngineModified(Cycle65_UnifiedPeakReversionEngine):
    def __init__(self, min_range_bars=10, min_rng_dollars=15.0):
        super().__init__(symbol="BTCUSDT")
        self.min_range_bars = min_range_bars
        self.min_rng_dollars = min_rng_dollars

    def _is_range_boundary_violation(self, primary_direction: int, mid: float) -> bool:
        if len(self.rolling_15m_prices) >= self.min_range_bars:
            sess_high = max(self.rolling_15m_prices)
            sess_low = min(self.rolling_15m_prices)
            rng = sess_high - sess_low
            if rng >= self.min_rng_dollars:
                pct = (mid - sess_low) / rng
                if primary_direction == 0 and pct <= 0.25:
                    return True
                if primary_direction == 2 and pct >= 0.75:
                    return True
        return False

for mb in [30, 15, 10, 8]:
    all_closed_trades = []
    losses = []
    long_trades = []
    short_trades = []

    for arch in archives:
        ledgers = glob.glob(os.path.join(base_dir, arch, '*ledger*.jsonl'))
        if not ledgers: continue
        with open(ledgers[0], 'r', encoding='utf-8') as f:
            bars = [json.loads(l) for l in f if l.strip().startswith('{')]

        engine = UnifiedEngineModified(min_range_bars=mb, min_rng_dollars=15.0)

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

        for t in engine.closed_trades:
            item = {'cycle': arch, 'side': t.side, 'net': t.net_return_bps, 'reason': t.exit_reason}
            all_closed_trades.append(item)
            if t.side == 'BUY': long_trades.append(item)
            else: short_trades.append(item)
            if t.net_return_bps <= 0: losses.append(item)

    n = len(all_closed_trades)
    w = sum(1 for t in all_closed_trades if t['net'] > 0)
    wr = (w / n * 100.0) if n > 0 else 0.0
    tot_net = sum(t['net'] for t in all_closed_trades)
    print(f"=== min_range_bars = {mb} ===")
    print(f"  Total Trades: {n} | Wins: {w} | Losses: {n-w} | WR: {wr:.2f}% | Net: {tot_net:+.2f} bps")
    print(f"  Losses: {[l['cycle'] + '_' + l['side'] for l in losses]}")
