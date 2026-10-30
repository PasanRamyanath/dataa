import json

with open("scratch/archive_cycle39/champion_challenger_final_results.json") as f:
    d = json.load(f)

c1_trades = d["competitor_trade_ledgers"]["Challenger_1"]
c2_trades = d["competitor_trade_ledgers"]["Challenger_2"]

print("--- CHALLENGER 1 (5 trades, 80% WR) ---")
for i, t in enumerate(c1_trades):
    print(f"Trade {i+1}: side={t['side']} entry={t['entry_price']} exit={t['exit_price']} gross={t['gross_return_bps']:.2f}b net={t['net_return_bps']:.2f}b dur={t['holding_duration_sec']}s reason={t['exit_reason']}")

print("\n--- CHALLENGER 2 (3 trades, 100% WR) ---")
for i, t in enumerate(c2_trades):
    print(f"Trade {i+1}: side={t['side']} entry={t['entry_price']} exit={t['exit_price']} gross={t['gross_return_bps']:.2f}b net={t['net_return_bps']:.2f}b dur={t['holding_duration_sec']}s reason={t['exit_reason']}")
