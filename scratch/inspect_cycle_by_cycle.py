import os
import sys

sys.path.insert(0, os.path.abspath('feature engineering scripts'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from simulate_all_59_cycles_qualification import run_full_simulation, Cycle35_ReversionPureScalper

archives = [d for d in os.listdir("scratch") if d.startswith("archive_cycle") and os.path.isdir(os.path.join("scratch", d))]
archives = sorted(archives, key=lambda x: int(x.replace("archive_cycle", "")))
valid = [d for d in archives if os.path.exists(os.path.join("scratch", d, "live_hierarchical_session_1h_ledger.jsonl")) and os.path.getsize(os.path.join("scratch", d, "live_hierarchical_session_1h_ledger.jsonl")) > 0]

res = run_full_simulation(lambda: Cycle35_ReversionPureScalper(symbol="BTCUSDT"), "Cycle 35 Champion", valid)
cr = res["cycle_results"]

print("\nCYCLE BY CYCLE BREAKDOWN:")
pos_cycles = 0
neg_cycles = 0
zero_cycles = 0
for c in valid:
    data = cr[c]
    trades = data["trades"]
    pnl = data["net_pnl_bps"]
    wr = data["win_rate"]
    tag = "WIN" if pnl > 0 else ("LOSS" if pnl < 0 else "FLAT")
    if pnl > 0: pos_cycles += 1
    elif pnl < 0: neg_cycles += 1
    else: zero_cycles += 1
    if trades > 0:
        print(f"{c:18s}: {trades:2d} trades | WR: {wr:5.1f}% | Net: {pnl:+6.2f} bps [{tag}]")

print(f"\nPositive Cycles: {pos_cycles} | Negative Cycles: {neg_cycles} | Zero/Flat: {zero_cycles}")
