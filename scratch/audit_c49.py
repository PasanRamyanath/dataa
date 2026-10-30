import json

with open('scratch/archive_cycle49/champion_challenger_final_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(json.dumps(data.get('competitor_trade_ledgers', {}).get('Champion', []), indent=2))
