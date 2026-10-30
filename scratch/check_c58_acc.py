import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('scratch/live_session_final_metrics.json', encoding='utf-8') as f:
    d = json.load(f)

print("Bars evaluated:", d.get('bars_evaluated'))
print("Duration:", d.get('session_duration_minutes'))
print("Accuracy summary:")
for h, acc in d.get('accuracy_summary', {}).items():
    if isinstance(acc, dict):
        print(f"  {h}: Overall={acc.get('overall_accuracy_pct')}% | Active Precision={acc.get('active_precision_pct')}% | Active Count={acc.get('active_signals_count')}")
