import os
import glob
import json

base_dir = "D:/PrivateProjects/BTC Futures/scratch"
archives = sorted([d for d in os.listdir(base_dir) if (d.startswith("archive_cycle") or d.startswith("archive_v3")) and os.path.isdir(os.path.join(base_dir, d))])

results = []

for arch in archives:
    res_path = os.path.join(base_dir, arch, "champion_challenger_final_results.json")
    lb_path = os.path.join(base_dir, arch, "champion_challenger_leaderboard.json")
    target = res_path if os.path.exists(res_path) else (lb_path if os.path.exists(lb_path) else None)
    if not target:
        continue
    
    try:
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        continue
    
    lb = data.get("final_leaderboard", data.get("leaderboard", []))
    if not lb:
        continue
    valid = [c for c in lb if c.get("trades", 0) > 0]
    if not valid:
        continue
    best = sorted(valid, key=lambda x: x.get("net_pnl_bps", -999), reverse=True)[0]
    results.append({
        "archive": arch,
        "winner": best.get("name"),
        "profile": best.get("strategy_profile", "")[:65],
        "trades": best.get("trades"),
        "wins": best.get("wins"),
        "wr": best.get("win_rate_pct"),
        "pnl": best.get("net_pnl_bps"),
        "avg": best.get("avg_trade_bps")
    })

print(f"Total archives with trades: {len(results)}")
sorted_pnl = sorted(results, key=lambda x: x["pnl"], reverse=True)
print("\n--- TOP 15 SESSIONS BY ACTUAL LIVE NET PNL ---")
for r in sorted_pnl[:15]:
    print(f"{r['archive']:<22} | {r['winner']:<12} | Tr: {r['trades']:<2} | WR: {r['wr']:>5.1f}% | Net: {r['pnl']:>7.2f} bps | Avg: {r['avg']:>6.2f} bps | {r['profile']}")

wr100 = [r for r in sorted_pnl if r["wr"] == 100.0]
print(f"\n--- SESSIONS WITH 100% LIVE WIN RATE ({len(wr100)} sessions) ---")
for r in wr100:
    print(f"{r['archive']:<22} | {r['winner']:<12} | Tr: {r['trades']:<2} | Net: {r['pnl']:>7.2f} bps | Avg: {r['avg']:>6.2f} bps | {r['profile']}")
