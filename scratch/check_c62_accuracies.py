import json

with open('scratch/live_session_final_metrics.json', 'r') as f:
    d = json.load(f)

print("FORWARD PREDICTION ACCURACIES & ACTIVE PRECISIONS (Cycle 62, 360 bars):")
for horizon, acc in d.get('accuracy_summary', {}).items():
    if isinstance(acc, dict):
        print(f"  {horizon:<6}: Overall Acc: {acc.get('overall_accuracy_pct', 0):.2f}% | Active Prec: {acc.get('active_precision_pct', 0):.2f}% (Signals: {acc.get('active_signals_count', 0)}/{acc.get('evaluated_samples', 0)})")
    else:
        print(f"  {horizon:<6}: {acc}")
