import json

with open('scratch/champion_challenger_final_results.json', 'r') as f:
    d = json.load(f)

for comp_name in ["Champion", "Challenger_1"]:
    trades = d.get('competitor_trade_ledgers', {}).get(comp_name, [])
    print(f"\n=================== {comp_name} TRADES ({len(trades)}) ===================")
    for i, t in enumerate(trades):
        print(f"Trade {i+1}: {t.get('side')} @ {t.get('entry_price')} -> {t.get('exit_price')} | Net: {t.get('net_return_bps'):+.2f}b | Gross: {t.get('gross_return_bps'):+.2f}b | Fee: {t.get('total_fee_bps')}b | MFE: {t.get('peak_mfe_bps'):+.2f}b | MAE: {t.get('max_mae_bps'):+.2f}b | Dur: {t.get('holding_duration_sec')}s | Exit: {t.get('exit_reason')}")
