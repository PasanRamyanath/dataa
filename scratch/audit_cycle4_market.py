import json
import numpy as np

bars = []
with open('scratch/live_hierarchical_session_1h_ledger.jsonl', 'r') as f:
    for line in f:
        bars.append(json.loads(line))

prices = [b["bbo_mid"] for b in bars]
p_open = prices[0]
p_close = prices[-1]
p_high = max(prices)
p_low = min(prices)
max_range_bps = ((p_high - p_low) / p_open) * 1e4
net_drift_bps = ((p_close - p_open) / p_open) * 1e4

print(f"Total Bars: {len(bars)}")
print(f"Open: ${p_open:.2f} | Close: ${p_close:.2f} | High: ${p_high:.2f} | Low: ${p_low:.2f}")
print(f"Total Range: {max_range_bps:.2f} bps (${p_high - p_low:.2f}) | Net Drift: {net_drift_bps:+.2f} bps")

# Check distribution of primary_direction and expected_net_edge_bps
dirs = [b["primary_direction"] for b in bars]
edges = [b["expected_net_edge_bps"] for b in bars]
modes = [b["execution_mode"] for b in bars]

print("\nDirection counts:", {0: dirs.count(0), 1: dirs.count(1), 2: dirs.count(2)})
print("Execution mode counts:", {m: modes.count(m) for m in set(modes)})
print(f"Max edge: {max(edges):.2f} bps | Min edge: {min(edges):.2f} bps | Mean edge: {np.mean(edges):.2f} bps")

# Check macro and meso probabilities across the session
macro_dn = [b["macro_probs"][0] for b in bars]
macro_up = [b["macro_probs"][2] for b in bars]
meso_dn = [b["meso_probs"][0] for b in bars]
meso_up = [b["meso_probs"][2] for b in bars]
tact_dn = [b["tactical_probs"][0] for b in bars]
tact_up = [b["tactical_probs"][2] for b in bars]

print(f"Macro P(Up) max: {max(macro_up):.3f}, P(Dn) max: {max(macro_dn):.3f}")
print(f"Meso  P(Up) max: {max(meso_up):.3f}, P(Dn) max: {max(meso_dn):.3f}")
print(f"Tact  P(Up) max: {max(tact_up):.3f}, P(Dn) max: {max(tact_dn):.3f}")
