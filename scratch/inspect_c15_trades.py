import json

with open("scratch/live_hierarchical_session_1h_ledger.jsonl", "r", encoding="utf-8") as f:
    bars = [json.loads(line) for line in f]

for b in bars:
    if b.get("closed_trades_count", 0) > 0:
        print(f"Trade 1 closed at Bar {b['bar_idx']:03d} | Mid: {b['bbo_mid']} | Time: {b['datetime']}")
        break
