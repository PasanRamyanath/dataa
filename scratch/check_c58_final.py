import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('scratch/champion_challenger_final_results.json', encoding='utf-8') as f:
    d = json.load(f)

print(json.dumps(d, indent=2))
