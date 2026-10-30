import json

with open('scratch/live_session_1h_heartbeat.json', 'r', encoding='utf-8') as f:
    hb = json.load(f)

print(f"Bar: {hb['bars_evaluated']} / 360 | Elapsed: {hb['elapsed_minutes']:.1f}m | Price: {hb['current_price']} | Champ Net PnL: {hb['net_pnl_bps']} bps")
