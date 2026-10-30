import json, sys
sys.stdout.reconfigure(encoding='utf-8')

for fname in ['scratch/live_session_final_metrics.json', 'scratch/live_session_1h_final_metrics.json']:
    try:
        with open(fname, encoding='utf-8') as f:
            d = json.load(f)
            print(f"=== {fname} ===")
            print("Session Duration Min:", d.get('session_duration_minutes'))
            print("Bars evaluated:", d.get('bars_evaluated'))
            print("Accuracy summary:")
            for h, acc in d.get('accuracy_summary', {}).items():
                if isinstance(acc, dict):
                    print(f"  {h}: Overall={acc.get('overall_accuracy_pct')}% | Active Precision={acc.get('active_precision_pct')}% | Active Count={acc.get('active_signals_count')}")
    except Exception as e:
        print(fname, e)
