"""
scratch/analyze_cycles_root_cause.py
Compares all closed trades across Cycles 25, 26, 27, 28, 29
to isolate the exact mechanism that turned winning sessions into losing sessions.
"""
import json
from pathlib import Path

cycles = [25, 26, 27, 28, 29]
base_dir = Path("scratch")

print("=" * 90)
print(f"{'Cycle':<8}{'Champion Net':<14}{'Winner Name':<18}{'Winner Net':<12}{'Winner WR':<12}{'Winner Trades':<14}")
print("-" * 90)

for c in cycles:
    res_path = base_dir / f"archive_cycle{c}" / "champion_challenger_final_results.json"
    if not res_path.exists():
        continue
    with open(res_path, "r") as f:
        data = json.load(f)
    champ = next((x for x in data["final_leaderboard"] if x["name"] == "Champion"), None)
    champ_net = f"{champ['net_pnl_bps']:+.2f} bps" if champ else "N/A"
    w_name = data.get("tournament_winner", "N/A")
    w_net = f"{data.get('winner_net_pnl_bps', 0.0):+.2f} bps"
    w_wr = f"{data.get('winner_win_rate_pct', 0.0):.1f}%"
    w_tr = str(data.get("winner_trades", 0))
    print(f"C{c:<7}{champ_net:<14}{w_name:<18}{w_net:<12}{w_wr:<12}{w_tr:<14}")

print("=" * 90)
print("\nBREAKDOWN OF EXIT REASONS FOR TOURNAMENT WINNERS:")
print("-" * 90)

for c in cycles:
    res_path = base_dir / f"archive_cycle{c}" / "champion_challenger_final_results.json"
    if not res_path.exists():
        continue
    with open(res_path, "r") as f:
        data = json.load(f)
    w_name = data.get("tournament_winner", "N/A")
    trades = data.get("competitor_trade_ledgers", {}).get(w_name, [])
    print(f"\n--- Cycle {c} Winner: {w_name} ({len(trades)} trades) ---")
    for i, t in enumerate(trades, 1):
        print(f"  Trade {i}: {t['side']:<4} @ {t['entry_price']:.1f} -> {t['exit_price']:.1f} | "
              f"Hold: {t['holding_duration_sec']:>4.0f}s | Gross: {t['gross_return_bps']:+6.2f} bps | "
              f"Net: {t['net_return_bps']:+6.2f} bps | MFE: {t['peak_mfe_bps']:+5.2f}b | Reason: {t['exit_reason']}")
