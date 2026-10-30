import json, sys

print("=== LIVE CYCLE 57 METRICS ===")
with open("scratch/archive_cycle57/live_session_final_metrics.json") as f:
    m57 = json.load(f)
print(f"Total trades: {m57.get('total_trades')}, Win Rate: {m57.get('win_rate_pct')}%, Net: {m57.get('total_net_pnl_bps')} bps")
for t in m57.get("closed_trades", []):
    print("  Live:", t.get("side"), t.get("entry_price"), "->", t.get("exit_price"), "Net:", t.get("net_return_bps"), t.get("exit_reason"))

print("\n=== LIVE CYCLE 57 TOURNAMENT RESULTS ===")
with open("scratch/archive_cycle57/champion_challenger_final_results.json") as f:
    t57 = json.load(f)
for comp, trades in t57.get("competitor_trade_ledgers", {}).items():
    print(f"[{comp}] ({len(trades)} trades):")
    for t in trades:
        print(f"  {t.get('side')} @ {t.get('entry_price')} -> {t.get('exit_price')} | Net: {t.get('net_return_bps')} bps | {t.get('exit_reason')}")
