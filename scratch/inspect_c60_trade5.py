import json

with open('scratch/archive_cycle60/champion_challenger_final_results.json', 'r') as f:
    d = json.load(f)

champ_trades = d.get('competitor_trades', {}).get('Champion', [])
print(f"Total Champion Trades in Cycle 60: {len(champ_trades)}")
for i, t in enumerate(champ_trades):
    print(f"\n--- Trade {i+1} ---")
    for k, v in t.items():
        print(f"  {k}: {v}")
