import json

with open('scratch/archive_cycle58/champion_challenger_final_results.json', 'r') as f:
    d = json.load(f)
trades = d.get('competitor_trade_ledgers', {}).get('Champion', [])
print(f"Trades for Champion in Cycle 58 live: {len(trades)}")
for t in trades:
    print(t)
