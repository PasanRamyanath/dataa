import sys, os, json
sys.path.insert(0, os.path.abspath('feature engineering scripts'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from testing.run_56_cycle_historical_qualification import run_56_cycle_battery
from execution.champion_challenger_framework import (
    Cycle35_ReversionPureScalper,
    Cycle35_TightRangeReversionScalper,
    Cycle60_ExhaustionAwareScalper,
)

print("=" * 80)
print("EXECUTING FULL 56-CYCLE HISTORICAL QUALIFICATION BATTERY")
print("=" * 80)

# Evaluate Champion Baseline
print("\n>>> Replaying Champion Baseline across ALL 56 Cycles...")
res_champ = run_56_cycle_battery(
    candidate_cls=Cycle35_ReversionPureScalper,
    candidate_name="Reigning Champion: Cycle 35 Reversion Pure Scalper",
    problem_addressed="Foundational Conformal Certainty Baseline"
)

print(f"Results across {res_champ['total_cycles']} Cycles:")
print(f"  Total Trades: {res_champ['total_trades']} ({res_champ['avg_trades_per_hour']} trades/hr)")
print(f"  Win Rate: {res_champ['overall_win_rate_pct']}% ({res_champ['total_wins']}W / {res_champ['total_losses']}L)")
print(f"  Total Net Realized PnL: {res_champ['total_net_pnl_bps']:+.2f} bps")
print(f"  Winning Cycles: {res_champ['winning_cycles']} | Break-even: {res_champ['break_even_cycles']} | Losing: {res_champ['losing_cycles']}")
print(f"  Passed Qualification Standard: {res_champ['passed_qualification']}")

# Save results
with open("scratch/56_cycle_champion_qualification_results.json", "w", encoding="utf-8") as f:
    json.dump(res_champ, f, indent=2)

print("\nSaved full 56-cycle results to scratch/56_cycle_champion_qualification_results.json")
