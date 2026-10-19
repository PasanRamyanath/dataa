import json
import numpy as np

LOG_PATH = "feature engineering scripts/scratch/monitor_30m_log.jsonl"

def load_bars():
    bars = []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                bars.append(json.loads(line.strip()))
    return bars

bars = load_bars()
n_bars = len(bars)
print(f"Loaded {n_bars} bars.")

# For each bar, we know:
# Future 3-minute return (18 bars ahead) and 5-minute return (30 bars ahead)
returns_10s = []
returns_30s = []
returns_1m = []
returns_3m = []
returns_5m = []

for i in range(n_bars):
    p0 = bars[i]["binance_price"]
    # 10s return (1 bar)
    r10 = ((bars[i+1]["binance_price"] - p0) / p0 * 10000.0) if (i + 1 < n_bars) else 0.0
    # 30s return (3 bars)
    r30 = ((bars[i+3]["binance_price"] - p0) / p0 * 10000.0) if (i + 3 < n_bars) else 0.0
    # 1m return (6 bars)
    r1m = ((bars[i+6]["binance_price"] - p0) / p0 * 10000.0) if (i + 6 < n_bars) else 0.0
    # 3m return (18 bars)
    r3m = ((bars[i+18]["binance_price"] - p0) / p0 * 10000.0) if (i + 18 < n_bars) else 0.0
    # 5m return (30 bars)
    r5m = ((bars[i+30]["binance_price"] - p0) / p0 * 10000.0) if (i + 30 < n_bars) else 0.0
    
    returns_10s.append(r10)
    returns_30s.append(r30)
    returns_1m.append(r1m)
    returns_3m.append(r3m)
    returns_5m.append(r5m)

members = list(bars[0]["members"].keys())

# Evaluate each member's correlation with forward returns
print("\n" + "=" * 85)
print("INDIVIDUAL MEMBER CORRELATION WITH FORWARD RETURNS:")
print(f"{'Member':<20} | {'Corr 10s':<10} | {'Corr 30s':<10} | {'Corr 1m':<10} | {'Corr 3m':<10} | {'Corr 5m':<10}")
print("-" * 85)

member_edges = {}
for m in members:
    edges = []
    for b in bars:
        p = b["members"][m]
        # Directional edge: P(Up) - P(Down)
        edges.append(p[2] - p[0])
    edges = np.array(edges)
    member_edges[m] = edges
    
    c10 = np.corrcoef(edges[:-1], returns_10s[:-1])[0, 1]
    c30 = np.corrcoef(edges[:-3], returns_30s[:-3])[0, 1]
    c1m = np.corrcoef(edges[:-6], returns_1m[:-6])[0, 1]
    c3m = np.corrcoef(edges[:-18], returns_3m[:-18])[0, 1]
    c5m = np.corrcoef(edges[:-30], returns_5m[:-30])[0, 1]
    print(f"{m:<20} | {c10:+8.4f}   | {c30:+8.4f}   | {c1m:+8.4f}   | {c3m:+8.4f}   | {c5m:+8.4f}")

# Now test Ensembling Methods
# 1. Equal-Weight Mean Member Edge
mean_edge = np.mean([member_edges[m] for m in members], axis=0)

# 2. Optimal Convex Weighted Ensemble (higher weight on top performing models: itransformer, nhits, gbm, tft)
# Rank by 3m correlation
weights = {}
tot_w = 0.0
for m in members:
    c = np.corrcoef(member_edges[m][:-18], returns_3m[:-18])[0, 1]
    w = max(0.01, c) # only positive weights
    weights[m] = w
    tot_w += w
for m in members: weights[m] /= tot_w

opt_edge = sum(weights[m] * member_edges[m] for m in members)

# 3. Current Live Fusion Edge (from bars: prob_up - prob_down)
live_edge = np.array([b["p_up"] - b["p_down"] for b in bars])

print("\n" + "=" * 85)
print("ENSEMBLE FUSION CORRELATIONS WITH FORWARD RETURNS:")
print(f"{'Fusion Scheme':<35} | {'Corr 30s':<10} | {'Corr 1m':<10} | {'Corr 3m':<10} | {'Corr 5m':<10}")
print("-" * 85)

for name, edge in [
    ("Equal-Weight Mean Vote", mean_edge),
    ("Information-Ratio Weighted Ensemble", opt_edge),
    ("Current Live Stacking+Bayesian Loop", live_edge)
]:
    c30 = np.corrcoef(edge[:-3], returns_30s[:-3])[0, 1]
    c1m = np.corrcoef(edge[:-6], returns_1m[:-6])[0, 1]
    c3m = np.corrcoef(edge[:-18], returns_3m[:-18])[0, 1]
    c5m = np.corrcoef(edge[:-30], returns_5m[:-30])[0, 1]
    print(f"{name:<35} | {c30:+8.4f}   | {c1m:+8.4f}   | {c3m:+8.4f}   | {c5m:+8.4f}")
