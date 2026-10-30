import sys, os, glob, json, types
sys.path.insert(0, 'feature engineering scripts')
from execution.champion_challenger_framework import Cycle66_QuadModeMultiTranchePeakEngine

base_dir = 'D:/PrivateProjects/BTC Futures/scratch'
archives = sorted([d for d in os.listdir(base_dir) if (d.startswith('archive_cycle') or d.startswith('archive_v3')) and os.path.isdir(os.path.join(base_dir, d))])

trade_records = []

for arch in archives:
    ledgers = glob.glob(os.path.join(base_dir, arch, '*ledger*.jsonl'))
    if not ledgers: continue
    bars = []
    with open(ledgers[0], 'r') as f:
        for line in f:
            if line.strip():
                try: bars.append(json.loads(line))
                except: pass
    if len(bars) < 30: continue

    engine = Cycle66_QuadModeMultiTranchePeakEngine('BTCUSDT')
    for idx, bar in enumerate(bars):
        mid = bar.get('bbo_mid', bar.get('mid', 0.0))
        if mid <= 0: continue
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

        w_before = getattr(engine, 'pos_weight', 1.0)
        n_closed_before = len(engine.closed_trades)
        engine.on_market_micro_tick(ts_ms, bid, ask, bid_qty, ask_qty, bar_high=ask+0.05, bar_low=bid-0.05, current_decision=decision)
        if len(engine.closed_trades) > n_closed_before:
            t = engine.closed_trades[-1]
            trade_records.append({'net_bps': t.net_return_bps, 'weight': w_before, 'reason': t.exit_reason})
            continue

        w_before = getattr(engine, 'pos_weight', 1.0)
        n_closed_before = len(engine.closed_trades)
        engine.on_hierarchical_decision(decision, bid, ask, bid_qty, ask_qty, equity_usdt=10000.0, intra_bar_open=mid)
        if len(engine.closed_trades) > n_closed_before:
            t = engine.closed_trades[-1]
            trade_records.append({'net_bps': t.net_return_bps, 'weight': w_before, 'reason': t.exit_reason})
            continue

        if engine.state == 'RESTING_ENTRY_QUEUE':
            engine.on_market_micro_tick(ts_ms + 1000, bid, ask, bid_qty, ask_qty, bar_high=ask+0.05, bar_low=bid-0.05, current_decision=decision)

    if engine.active_position is not None:
        last_mid = bars[-1].get('bbo_mid', bars[-1].get('mid', 0.0))
        pos = engine.active_position
        sign = 1.0 if pos.side == 'BUY' else -1.0
        avg_ep = getattr(engine, 'pos_avg_entry_price', pos.entry_price) or pos.entry_price
        u_bps = sign * ((last_mid - avg_ep) / avg_ep) * 1e4
        w_final = getattr(engine, 'pos_weight', 1.0)
        engine._close_position(exit_price=last_mid, exit_ts_ms=bars[-1].get('ts_ms', 3600000), exit_fee_bps=2.0, reason='SESSION_END_MARK_TO_MARKET')
        t = engine.closed_trades[-1]
        trade_records.append({'net_bps': t.net_return_bps, 'weight': w_final, 'reason': t.exit_reason})

initial_capital = 600.0
target_capital = 6000.0
leverage = 25.0
base_frac = leverage / 15.0

hours_in_dataset = 64.0
trades_per_hour = len(trade_records) / hours_in_dataset
trades_per_day = trades_per_hour * 24.0

print(f"Dataset Trades: {len(trade_records)} across 64 hours ({trades_per_hour:.2f} trades/hr, {trades_per_day:.1f} trades/day)")

def simulate_to_target(records, friction_bps=0.0, daily_funding=0.0):
    cap = initial_capital
    trade_idx = 0
    total_trades = 0
    day = 0
    day_log = []
    while cap < target_capital and day < 120:
        day += 1
        trades_today = int(round(trades_per_day))
        for _ in range(trades_today):
            r = records[trade_idx % len(records)]
            trade_idx += 1
            total_trades += 1
            net_bps = r['net_bps'] - friction_bps
            eff_lev = r['weight'] * base_frac
            ret_pct = eff_lev * (net_bps / 10000.0)
            cap *= (1.0 + ret_pct)
            if cap >= target_capital:
                break
        if daily_funding > 0:
            cap *= (1.0 - daily_funding)
        day_log.append((day, cap, total_trades))
        if cap >= target_capital:
            break
    return day, total_trades, cap, day_log

# Case 1: Pure Empirical
d1, t1, c1, log1 = simulate_to_target(trade_records, friction_bps=0.0, daily_funding=0.0)
print(f"\nCase 1 (Pure Empirical): Reached ${c1:.2f} in {d1} days ({t1} trades)")

# Case 2: True 24/7 Continuous (no session cuts)
trade_records_cont = [r for r in trade_records if r['reason'] != 'SESSION_END_MARK_TO_MARKET']
d2, t2, c2, log2 = simulate_to_target(trade_records_cont, friction_bps=0.0, daily_funding=0.0)
print(f"Case 2 (Continuous 24/7): Reached ${c2:.2f} in {d2} days ({t2} trades)")

# Case 3: Prudent Real-World Friction
d3, t3, c3, log3 = simulate_to_target(trade_records, friction_bps=0.50, daily_funding=0.0015)
print(f"Case 3 (Prudent Real-World): Reached ${c3:.2f} in {d3} days ({t3} trades)")

# Output key milestone snapshots for Case 3
print("\n--- PRUDENT REAL-WORLD MILESTONES (CASE 3) ---")
for milestone in [1000, 1500, 2000, 3000, 4000, 5000, 6000]:
    for day, cap, tr in log3:
        if cap >= milestone:
            print(f"Target ${milestone:>4d} USDT: Day {day:2d} ({tr:>4d} trades) -> Capital: ${cap:>7.2f} USDT")
            break
