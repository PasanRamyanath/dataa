import json

with open("scratch/live_hierarchical_session_1h_ledger.jsonl", "r", encoding="utf-8") as f:
    bars = [json.loads(line) for line in f]

closed = []
prev_cnt = 0
for b in bars:
    cnt = b.get("closed_trades_count", 0)
    if cnt > prev_cnt:
        closed.append((cnt, b["bar_idx"], b["ts_ms"], b["datetime"], b["bbo_mid"]))
        prev_cnt = cnt

print("=== CLOSED TRADES SO FAR ===")
for c in closed:
    print(f"Trade {c[0]}: Bar {c[1]:03d} | Time: {c[3]} | Mid: {c[4]:.2f}")
