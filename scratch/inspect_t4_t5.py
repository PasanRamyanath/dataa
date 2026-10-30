import json

bars = [json.loads(l) for l in open('scratch/live_hierarchical_session_1h_ledger.jsonl')]

for i in range(180, 260):
    cur_pos = bars[i].get('active_position')
    prev_pos = bars[i-1].get('active_position') if i > 0 else None
    if cur_pos is not None and prev_pos is None:
        print(f"Entry at Bar {i} (ts={bars[i]['ts_ms']}): side={cur_pos['side']}, entry={cur_pos['entry_price']:.2f}")
    if cur_pos is None and prev_pos is not None:
        exit_p = bars[i]['bbo_mid']
        sign = 1.0 if prev_pos['side'] == 'BUY' else -1.0
        gross = sign * (exit_p - prev_pos['entry_price']) / prev_pos['entry_price'] * 1e4
        print(f"Exit at Bar {i} (ts={bars[i]['ts_ms']}): side={prev_pos['side']}, entry={prev_pos['entry_price']:.2f}, exit={exit_p:.2f}, gross={gross:+.2f} bps, net={gross-4.0:+.2f} bps, peak={prev_pos['peak_favorable_excursion_bps']:.2f} bps")
