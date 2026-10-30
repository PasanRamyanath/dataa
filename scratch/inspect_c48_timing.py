import json

fpath = "scratch/archive_cycle48/live_hierarchical_session_1h_ledger.jsonl"
bars = []
with open(fpath, 'r', encoding='utf-8') as f:
    for line in f:
        bars.append(json.loads(line))

print(f"Loaded {len(bars)} bars from Cycle 48")

# Let's inspect when the single trade happened
for b in bars:
    if b.get('active_position') is not None or b.get('closed_trades_count', 0) > 0:
        print(f"Bar {b['bar_idx']} | Price: {b['bbo_mid']} | Pos: {b.get('active_position') is not None} | Closed: {b.get('closed_trades_count')}")
