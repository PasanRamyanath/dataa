import json

with open("scratch/champion_challenger_final_results.json") as f:
    d = json.load(f)

for k, v in d["competitor_trade_ledgers"].items():
    print(f"{k}: {len(v)} trades")
