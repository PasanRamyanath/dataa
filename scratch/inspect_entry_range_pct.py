import sys, os, glob, json
sys.path.insert(0, os.path.abspath('feature engineering scripts'))
sys.path.insert(0, os.path.abspath('.'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from scratch.test_quad_resonance_56 import QuadResonanceStrictScalper, all_ledgers, get_cycle_num, RealisticLedgerReplayer

trades_with_range = []

for l in all_ledgers:
    c_num = get_cycle_num(l)
    rep = RealisticLedgerReplayer(l)
    eng = QuadResonanceStrictScalper(symbol='BTCUSDT')
    trades = rep.replay(eng)
    
    records = rep.records
    ts_to_idx = {r['ts_ms']: idx for idx, r in enumerate(records)}
    
    for t in trades:
        entry_ts = t.entry_ts_ms
        idx = ts_to_idx.get(entry_ts)
        if idx is not None and idx >= 15:
            # Look at past 90 bars (15 minutes)
            lookback = min(90, idx)
            past_mids = [records[idx - k]['bbo_mid'] for k in range(lookback)]
            range_high = max(past_mids)
            range_low = min(past_mids)
            rng = range_high - range_low
            mid = t.entry_price
            range_pct = (mid - range_low) / rng if rng > 0 else 0.5
            
            trades_with_range.append({
                'cycle': c_num,
                'side': t.side,
                'net_pnl': t.net_return_bps,
                'is_win': t.net_return_bps > 0,
                'range_pct': range_pct,
                'mfe': t.peak_mfe_bps,
                'mae': t.max_mae_bps,
            })

wins = [x for x in trades_with_range if x['is_win']]
losses = [x for x in trades_with_range if not x['is_win']]

print(f"Total evaluated trades: {len(trades_with_range)}")
print(f"Wins: {len(wins)}, Losses: {len(losses)}")

print("\n--- RANGE POSITION AT ENTRY (0% = Range Low, 100% = Range High) ---")
avg_win_buy_range = sum(x['range_pct'] for x in wins if x['side'] == 'BUY') / max(1, sum(1 for x in wins if x['side'] == 'BUY'))
avg_loss_buy_range = sum(x['range_pct'] for x in losses if x['side'] == 'BUY') / max(1, sum(1 for x in losses if x['side'] == 'BUY'))
print(f"BUY Wins Avg Range Position:  {avg_win_buy_range*100:.1f}%")
print(f"BUY Losses Avg Range Position: {avg_loss_buy_range*100:.1f}%")

avg_win_sell_range = sum(x['range_pct'] for x in wins if x['side'] == 'SELL') / max(1, sum(1 for x in wins if x['side'] == 'SELL'))
avg_loss_sell_range = sum(x['range_pct'] for x in losses if x['side'] == 'SELL') / max(1, sum(1 for x in losses if x['side'] == 'SELL'))
print(f"SELL Wins Avg Range Position:  {avg_win_sell_range*100:.1f}%")
print(f"SELL Losses Avg Range Position: {avg_loss_sell_range*100:.1f}%")

# Let's inspect win rate by range deciles
print("\nBUY Win Rate by Range Position:")
for low_pct in range(0, 100, 20):
    high_pct = low_pct + 20
    bucket = [x for x in trades_with_range if x['side'] == 'BUY' and low_pct <= x['range_pct']*100 < high_pct]
    if bucket:
        b_wins = sum(1 for x in bucket if x['is_win'])
        wr = b_wins / len(bucket) * 100
        pnl = sum(x['net_pnl'] for x in bucket)
        print(f"  Range [{low_pct:2d}% - {high_pct:2d}%]: {len(bucket):2d} trades | Win Rate: {wr:5.1f}% | Net PnL: {pnl:+7.2f} bps")

print("\nSELL Win Rate by Range Position:")
for low_pct in range(0, 100, 20):
    high_pct = low_pct + 20
    bucket = [x for x in trades_with_range if x['side'] == 'SELL' and low_pct <= x['range_pct']*100 < high_pct]
    if bucket:
        b_wins = sum(1 for x in bucket if x['is_win'])
        wr = b_wins / len(bucket) * 100
        pnl = sum(x['net_pnl'] for x in bucket)
        print(f"  Range [{low_pct:2d}% - {high_pct:2d}%]: {len(bucket):2d} trades | Win Rate: {wr:5.1f}% | Net PnL: {pnl:+7.2f} bps")
