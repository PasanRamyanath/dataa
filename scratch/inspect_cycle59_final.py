import json
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

print("=== FINAL TOURNAMENT RESULTS ===")
try:
    with open("scratch/champion_challenger_final_results.json", "r") as f:
        res = json.load(f)
    print(json.dumps(res, indent=2))
except Exception as e:
    print(f"Error reading tournament results: {e}")

print("\n=== LIVE SESSION FINAL METRICS ===")
try:
    with open("scratch/live_session_final_metrics.json", "r") as f:
        met = json.load(f)
    print(json.dumps(met, indent=2))
except Exception as e:
    print(f"Error reading final metrics: {e}")
