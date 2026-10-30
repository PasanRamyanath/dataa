import json
from pathlib import Path

METRICS_PATH = Path(r"D:\PrivateProjects\BTC Futures\scratch\live_session_final_metrics.json")
LEDGER_PATH = Path(r"D:\PrivateProjects\BTC Futures\scratch\live_hierarchical_session_1h_ledger.jsonl")

if METRICS_PATH.exists():
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    print("=== FINAL METRICS ===")
    print(json.dumps(metrics, indent=2))
else:
    print("Metrics path does not exist!")

with open(LEDGER_PATH, "r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

print(f"\nTotal bars in ledger: {len(rows)}")

# Find all trade transitions
trades_seen = []
for i in range(1, len(rows)):
    prev_ap = rows[i-1].get('active_position')
    curr_ap = rows[i].get('active_position')
    
    # Entry
    if prev_ap is None and curr_ap is not None:
        trades_seen.append({
            'type': 'ENTRY',
            'bar': rows[i]['bar_idx'],
            'time': rows[i]['datetime'],
            'side': curr_ap['side'],
            'price': curr_ap['entry_price'],
            'ap': curr_ap
        })
    # Exit
    elif prev_ap is not None and curr_ap is None:
        trades_seen.append({
            'type': 'EXIT',
            'bar': rows[i]['bar_idx'],
            'time': rows[i]['datetime'],
            'exit_price': rows[i]['bbo_mid'],
            'prev_ap': prev_ap
        })

print(f"\nTrade Transitions ({len(trades_seen)} events):")
for t in trades_seen:
    print(t)
