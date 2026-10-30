import json

for c in [57, 58, 59]:
    p = f"scratch/archive_cycle{c}/champion_challenger_final_results.json"
    try:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
            lb = d.get("final_leaderboard", [])
            print(f"=== Cycle {c} ===")
            for r in lb[:4]:
                name = r.get("name")
                trades = r.get("trades")
                wr = r.get("win_rate_pct")
                pnl = r.get("net_pnl_bps")
                prof = r.get("strategy_profile")
                print(f"  {name:14s}: {trades} trades, {wr}% WR, {pnl:+.2f} bps | {prof}")
    except Exception as e:
        print(f"Cycle {c} error:", e)
