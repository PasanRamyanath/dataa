import sys, os, glob, json
sys.path.insert(0, os.path.abspath('feature engineering scripts'))
sys.path.insert(0, os.path.abspath('.'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from scratch.test_quad_resonance_56 import QuadResonanceStrictScalper, all_ledgers, get_cycle_num, RealisticLedgerReplayer

post_exit_recoveries = []

for l in all_ledgers:
    c_num = get_cycle_num(l)
    rep = RealisticLedgerReplayer(l)
    eng = QuadResonanceStrictScalper(symbol='BTCUSDT')
    trades = rep.replay(eng)
    
    # We also have rep.records (all bars in this cycle)
    records = rep.records
    ts_to_idx = {r['ts_ms']: idx for idx, r in enumerate(records)}
    
    for t in trades:
        if t.exit_reason == "RESPONSIVE_MODEL_MAKER_EXIT":
            exit_ts = t.exit_ts_ms
            idx = ts_to_idx.get(exit_ts)
            if idx is not None and idx + 30 < len(records):
                exit_p = t.exit_price
                future_mids = [records[idx + k]['bbo_mid'] for k in range(1, 31)]
                sign = 1.0 if t.side == "BUY" else -1.0
                
                # Check if price recovered in our original direction
                future_favorable = max(sign * (p - exit_p) for p in future_mids)
                future_favorable_bps = (future_favorable / exit_p) * 1e4
                post_exit_recoveries.append({
                    'cycle': c_num,
                    'side': t.side,
                    'net_loss': t.net_return_bps,
                    'future_favorable_bps': future_favorable_bps
                })

print(f"Analyzed {len(post_exit_recoveries)} RESPONSIVE_MODEL_MAKER_EXIT cuts:")
recovered_positive = sum(1 for x in post_exit_recoveries if x['future_favorable_bps'] >= 8.0)
recovered_5 = sum(1 for x in post_exit_recoveries if x['future_favorable_bps'] >= 5.0)
print(f"  Recovered >= +5.0 bps after cut: {recovered_5} / {len(post_exit_recoveries)} ({recovered_5/len(post_exit_recoveries)*100:.1f}%)")
print(f"  Recovered >= +8.0 bps after cut: {recovered_positive} / {len(post_exit_recoveries)} ({recovered_positive/len(post_exit_recoveries)*100:.1f}%)")
avg_rec = sum(x['future_favorable_bps'] for x in post_exit_recoveries) / len(post_exit_recoveries)
print(f"  Average post-exit favorable move: {avg_rec:+.2f} bps")
