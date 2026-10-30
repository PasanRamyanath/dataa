import json

with open("scratch/archive_cycle15/live_hierarchical_session_1h_ledger.jsonl", "r", encoding="utf-8") as f:
    bars = [json.loads(line) for line in f]

prices = [b["bbo_mid"] for b in bars]
print(f"Start price: {prices[0]:.2f}")
print(f"High price:  {max(prices):.2f} (bar {prices.index(max(prices)) + 1})")
print(f"Low price:   {min(prices):.2f} (bar {prices.index(min(prices)) + 1})")
print(f"End price:   {prices[-1]:.2f}")
print(f"Range:       {max(prices) - min(prices):.2f} USDT ({ (max(prices) - min(prices)) / prices[0] * 10000:.1f} bps)")
