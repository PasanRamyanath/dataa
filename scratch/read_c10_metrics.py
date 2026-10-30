import json

with open("scratch/archive_cycle10/live_session_1h_final_metrics.json", "r", encoding="utf-8") as f:
    m = json.load(f)

for i, t in enumerate(m["closed_trades"]):
    side = t["side"]
    net = t["net_return_bps"]
    reason = t["exit_reason"]
    dur = t["holding_duration_sec"]
    entry = t["entry_price"]
    exit_p = t["exit_price"]
    print(f"Cycle 10 Trade {i+1}: Side={side} Entry={entry} Exit={exit_p} Net={net:+.2f} bps ({reason}) dur={dur}s")
