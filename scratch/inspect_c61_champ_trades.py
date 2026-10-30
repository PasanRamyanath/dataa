import json

with open('scratch/champion_challenger_final_results.json', 'r') as f:
    d = json.load(f)

champ_trades = d.get('competitor_trade_ledgers', {}).get('Champion', [])
print(f"Total Champion Trades in Cycle 61: {len(champ_trades)}")
for i, t in enumerate(champ_trades):
    print(f"\n--- Trade {i+1} ---")
    print(f"  Side: {t.get('side')} | Entry: {t.get('entry_price')} -> Exit: {t.get('exit_price')}")
    print(f"  Gross: {t.get('gross_return_bps'):+.2f}b | Fees: {t.get('total_fee_bps'):.1f}b | Net: {t.get('net_return_bps'):+.2f}b")
    print(f"  MFE: {t.get('peak_mfe_bps'):+.2f}b | MAE: {t.get('max_mae_bps'):+.2f}b | Duration: {t.get('holding_duration_sec')}s")
    print(f"  Exit Reason: {t.get('exit_reason')}")
