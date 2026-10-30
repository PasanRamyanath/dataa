import json

with open('scratch/archive_cycle60/champion_challenger_final_results.json', 'r') as f:
    d = json.load(f)
lb = d.get('final_leaderboard', [])
if lb:
    print("Keys in leaderboard row:", list(lb[0].keys()))
    for row in lb:
        print(row)
