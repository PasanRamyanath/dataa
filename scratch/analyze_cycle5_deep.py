import json
import numpy as np

bars = [json.loads(l) for l in open('scratch/archive_cycle5/live_hierarchical_session_1h_ledger.jsonl')]

print(f"Total bars in Cycle 5: {len(bars)}")
prices = [b['bbo_mid'] for b in bars]
print(f"Price: start={prices[0]:.2f}, min={min(prices):.2f}, max={max(prices):.2f}, end={prices[-1]:.2f}")

# Look at trade 1 entry (Bar 139)
b139 = bars[139]
print("\nBar 139 (Trade 1 Entry):")
print(f"  mid: {b139['bbo_mid']:.2f}")
print(f"  macro: {b139['macro_probs']}, meso: {b139['meso_probs']}, tac: {b139['tactical_probs']}, mic: {b139['micro_probs']}")
print(f"  dir: {b139['primary_direction']}, edge: {b139['expected_net_edge_bps']}")

# Look at trade 2 entry (Bar 204)
b204 = bars[204]
print("\nBar 204 (Trade 2 Entry):")
print(f"  mid: {b204['bbo_mid']:.2f}")
print(f"  macro: {b204['macro_probs']}, meso: {b204['meso_probs']}, tac: {b204['tactical_probs']}, mic: {b204['micro_probs']}")
print(f"  dir: {b204['primary_direction']}, edge: {b204['expected_net_edge_bps']}")

# Look at trade 3 entry (Bar 215)
for i in range(210, 240):
    if bars[i].get('active_position') is not None and bars[i]['active_position']['side'] == 'BUY':
        b_buy = bars[i]
        print(f"\nTrade 3 First Seen at Bar {i}:")
        print(f"  mid: {b_buy['bbo_mid']:.2f}, entry: {b_buy['active_position']['entry_price']:.2f}")
        print(f"  macro: {b_buy['macro_probs']}, meso: {b_buy['meso_probs']}, tac: {b_buy['tactical_probs']}, mic: {b_buy['micro_probs']}")
        break
