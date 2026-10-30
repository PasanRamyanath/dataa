import json

prices = []
with open('scratch/live_hierarchical_session_1h_ledger.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        prices.append((data['bar_idx'], data['bbo_mid']))

print(f"Total bars recorded: {len(prices)}")
print(f"Start price: {prices[0][1]} @ Bar {prices[0][0]}")
print(f"Min price: {min(p[1] for p in prices)}")
print(f"Max price: {max(p[1] for p in prices)}")
print(f"Current price: {prices[-1][1]} @ Bar {prices[-1][0]}")

# Sample every 20 bars
print("\nPrice path:")
for idx, p in prices[::20]:
    print(f"  Bar {idx:03d}: ${p:.2f}")
print(f"  Bar {prices[-1][0]:03d}: ${prices[-1][1]:.2f}")
