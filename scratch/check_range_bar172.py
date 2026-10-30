import json

prices = []
with open("scratch/live_hierarchical_session_1h_ledger.jsonl") as f:
    for line in f:
        r = json.loads(line)
        if 82 <= r["bar_idx"] <= 172:
            prices.append(r["bbo_mid"])

print(f"Bars 82-172: min={min(prices):.2f}, max={max(prices):.2f}, range={max(prices)-min(prices):.2f}")
print(f"Bar 172 mid={prices[-1]:.2f}")
print(f"Distance from min: {prices[-1] - min(prices):.2f}")
