import json

with open('scratch/champion_challenger_final_results.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

champ_trades = d.get('competitor_trade_ledgers', {}).get('Champion', [])
print(f"Total Champion Trades in Cycle 60: {len(champ_trades)}")
for idx, t in enumerate(champ_trades, 1):
    side = t.get('side')
    entry = t.get('entry_price')
    exit_p = t.get('exit_price')
    net = t.get('net_return_bps')
    gross = t.get('gross_return_bps')
    fee = t.get('total_fee_bps')
    mfe = t.get('peak_mfe_bps')
    mae = t.get('max_mae_bps')
    dur = t.get('holding_duration_sec')
    reason = t.get('exit_reason')
    print(f"Trade {idx}: {side:4s} @ {entry:.1f} -> {exit_p:.1f} | Gross: {gross:+6.2f}b | Fee: {fee:.1f}b | Net: {net:+6.2f}b | MFE: {mfe:5.2f}b | MAE: {mae:5.2f}b | Dur: {dur:3.0f}s | {reason}")
