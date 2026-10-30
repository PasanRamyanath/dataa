import json, sys
sys.stdout.reconfigure(encoding='utf-8')

for fname in ['scratch/live_session_1h_heartbeat.json', 'scratch/champion_challenger_leaderboard.json']:
    try:
        with open(fname, encoding='utf-8') as f:
            d = json.load(f)
            print(f"=== {fname} ===")
            print(json.dumps(d, indent=2))
    except Exception as e:
        print(fname, e)
