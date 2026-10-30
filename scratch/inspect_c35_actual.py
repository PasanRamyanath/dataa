import json

with open("scratch/archive_cycle35/champion_challenger_final_results.json", "r") as f:
    res = json.load(f)

print("=== ACTUAL CYCLE 35 RESULTS ===")
print("Winner:", res.get("tournament_winner"), "Profile:", res.get("winner_profile"))
print("Net PnL:", res.get("winner_net_pnl_bps"), "WR:", res.get("winner_win_rate_pct"))
print("Actual trades in Cycle 35:")
for comp, trades in res.get("competitor_trade_ledgers", {}).items():
    if trades:
        print(f"[{comp}] {len(trades)} trades:")
        for t in trades:
            print(f"  {t.get('side')} @ {t.get('entry_price')} -> {t.get('exit_price')} | Net: {t.get('net_return_bps')} bps | {t.get('exit_reason')}")
