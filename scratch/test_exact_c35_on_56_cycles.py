import subprocess, sys, os, json
sys.path.insert(0, os.path.abspath('feature engineering scripts'))
sys.path.insert(0, os.path.abspath('.'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Extract exact champion_challenger_framework.py from commit 37b3b7d4
exact_code = subprocess.check_output(['git', 'show', '37b3b7d4:feature engineering scripts/execution/champion_challenger_framework.py'], encoding='utf-8')
with open('scratch/exact_c35_framework.py', 'w', encoding='utf-8') as f:
    f.write(exact_code)

from scratch.exact_c35_framework import Challenger5_ConformalConfidenceFilterScalper
from testing.run_56_cycle_historical_qualification import run_56_cycle_battery

print("=" * 80)
print("TESTING EXACT UNTOUCHED CYCLE 35 CHAMPION ACROSS ALL 56 CYCLES")
print("=" * 80)

res = run_56_cycle_battery(
    candidate_cls=Challenger5_ConformalConfidenceFilterScalper,
    candidate_name="Exact Cycle 35 Champion (Commit 37b3b7d4)",
    problem_addressed="Untouched Historical Gold Standard"
)

print(f"\nRESULTS ACROSS {res['total_cycles']} HISTORICAL CYCLES:")
print(f"  Total Trades: {res['total_trades']} ({res['avg_trades_per_hour']} trades/hr)")
print(f"  Win Rate: {res['overall_win_rate_pct']}% ({res['total_wins']}W / {res['total_losses']}L)")
print(f"  Total Net Realized PnL: {res['total_net_pnl_bps']:+.2f} bps")
print(f"  Winning Cycles: {res['winning_cycles']} | Break-even: {res['break_even_cycles']} | Losing: {res['losing_cycles']}")
print(f"  Passed Qualification Standard: {res['passed_qualification']}")

with open("scratch/56_cycle_exact_c35_results.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2)
