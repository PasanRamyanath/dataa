import json
with open('scratch/live_session_1h_heartbeat.json') as f:
    d = json.load(f)
print("Bars evaluated:", d.get('bars_evaluated'))
print("Current price:", d.get('current_price'))
print("Active pos:", d.get('active_position'))
print("Leaderboard summary:", json.dumps(d.get('leaderboard_summary'), indent=2))
