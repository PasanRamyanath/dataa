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
print(f"Total bars: {len(bars)}")

# Examine members
members_names = list(bars[0]["members"].keys())
print("Members:", members_names)

# Evaluate each member's directional calls:
# Class 0: Down (p[0] > p[2] and p[0] > p[1])
# Class 1: Flat (p[1] > p[0] and p[1] > p[2])
# Class 2: Up (p[2] > p[0] and p[2] > p[1])

member_stats = {m: {"down": 0, "flat": 0, "up": 0, "mean_p_down": [], "mean_p_flat": [], "mean_p_up": []} for m in members_names}

for b in bars:
    m_dict = b.get("members", {})
    for m, prob in m_dict.items():
        if m in member_stats:
            p = prob
            member_stats[m]["mean_p_down"].append(p[0])
            member_stats[m]["mean_p_flat"].append(p[1])
            member_stats[m]["mean_p_up"].append(p[2])
            call = np.argmax(p)
            if call == 0: member_stats[m]["down"] += 1
            elif call == 1: member_stats[m]["flat"] += 1
            elif call == 2: member_stats[m]["up"] += 1

print("\n" + "=" * 80)
print(f"{'Member':<20} | {'Calls (Down/Flat/Up)':<22} | {'Avg P(Down)':<12} | {'Avg P(Flat)':<12} | {'Avg P(Up)':<12}")
print("-" * 80)
for m in members_names:
    s = member_stats[m]
    n = len(bars)
    avg_d = np.mean(s["mean_p_down"])
    avg_f = np.mean(s["mean_p_flat"])
    avg_u = np.mean(s["mean_p_up"])
    calls = f"{s['down']}/{s['flat']}/{s['up']}"
    print(f"{m:<20} | {calls:<22} | {avg_d:10.4f}   | {avg_f:10.4f}   | {avg_u:10.4f}")
