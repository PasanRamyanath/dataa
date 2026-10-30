import json

fpath = "scratch/archive_cycle48/live_hierarchical_session_1h_ledger.jsonl"
bars = []
with open(fpath, 'r', encoding='utf-8') as f:
    for line in f:
        bars.append(json.loads(line))

# Check Bar 50 forward price action
b50 = bars[49] # 0-indexed idx 50
p_entry = b50['bbo_ask']
print(f"Bar 50 Entry Price: {p_entry}")
for i in range(50, min(100, len(bars))):
    b = bars[i]
    gross_bps = ((b['bbo_mid'] - p_entry) / p_entry) * 1e4
    net_bps = gross_bps - 4.0
    if gross_bps >= 15.0 or i == 99:
        print(f"Bar {b['bar_idx']:03d} (+{(i-49)*10}s) | Price: {b['bbo_mid']} | Gross: {gross_bps:+.2f} bps | Net: {net_bps:+.2f} bps")
        if gross_bps >= 15.0:
            break
