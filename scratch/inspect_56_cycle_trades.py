import json, sys

with open("scratch/56_cycle_true_peak_qualification_results.json", "r", encoding="utf-8") as f:
    res = json.load(f)

print("Winning Cycles (count={}):".format(res["winning_cycles"]))
for c in res["cycle_summaries"]:
    if c["net_pnl"] > 0:
        print(f"  Cycle {c['cycle']:2d}: {c['trades']} tr | {c['wins']}W/{c['losses']}L ({c['win_rate']}%) | Net: {c['net_pnl']:+.2f} bps")

print("\nWorst 5 Losing Cycles:")
losing = sorted([c for c in res["cycle_summaries"] if c["net_pnl"] < 0], key=lambda x: x["net_pnl"])
for c in losing[:5]:
    print(f"  Cycle {c['cycle']:2d}: {c['trades']} tr | {c['wins']}W/{c['losses']}L ({c['win_rate']}%) | Net: {c['net_pnl']:+.2f} bps")
