import sys, os, json
sys.path.insert(0, os.path.abspath('feature engineering scripts'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from testing.challenger_historical_qualification_harness import (
    evaluate_candidate_challenger,
    HistoricalLedgerReplayer
)
from execution.champion_challenger_framework import (
    Cycle35_ReversionPureScalper,
    Cycle35_TightRangeReversionScalper,
    Cycle60_ExhaustionAwareScalper,
    Cycle60_ActiveFrequencyScalper,
    Cycle60_SteppedRatchetScalper,
    Cycle60_DynamicAsymmetricScalper,
)

# Test Battery of Historical Cycles spanning all market regimes:
benchmark_cycles = [
    "archive_cycle12",  # Quiet low-vol Maker win (+3.52 bps, 100% WR)
    "archive_cycle17",  # Vertical +72 bps parabolic short squeeze
    "archive_cycle35",  # Baseline conformal benchmark (+44.64 bps, 100% WR)
    "archive_cycle48",  # High-frequency impulse breakout session
    "archive_cycle57",  # Fleet breakout + hollow-depth pullback trap
    "archive_cycle58",  # Sudden -$437 liquidation drop
    "archive_cycle59",  # Range breakdown + $6 envelope discovery
]

print("=" * 80)
print("RUNNING MULTI-CYCLE HISTORICAL QUALIFICATION BATTERY FOR CHALLENGERS")
print(f"Historical Benchmarks: {', '.join(benchmark_cycles)}")
print("=" * 80)

candidates = [
    {
        "class": Cycle35_ReversionPureScalper,
        "name": "Champion Benchmark (Cycle 35 Baseline)",
        "problem": "Foundational Conformal Certainty Baseline"
    },
    {
        "class": Cycle35_TightRangeReversionScalper,
        "name": "Challenger 1 Candidate ($6 Range Envelope Scalper)",
        "problem": "Solves range midpoint whipsaw by entering within $6 of 15m extremes"
    },
    {
        "class": Cycle60_ExhaustionAwareScalper,
        "name": "Challenger 2 Candidate (Exhaustion-Aware $6 Envelope Scalper)",
        "problem": "Solves tail-end chasing by vetoing entries in bottom/top 20% of extended range"
    },
    {
        "class": Cycle60_ActiveFrequencyScalper,
        "name": "Challenger 3 Candidate (Active Frequency $5 Envelope Scalper)",
        "problem": "Solves low-frequency/overly conservative trade starvation"
    },
    {
        "class": Cycle60_SteppedRatchetScalper,
        "name": "Challenger 4 Candidate (Stepped Ratchet $6 Envelope Scalper)",
        "problem": "Solves profit decay by locking +1.5b net at +4.5b and +3.5b at +7.5b MFE"
    },
    {
        "class": Cycle60_DynamicAsymmetricScalper,
        "name": "Challenger 5 Candidate (Dynamic Asymmetric Scalper)",
        "problem": "Combines $6 envelope, exhaustion guard, and micro-trailing profit protection"
    },
]

reports = []
for c in candidates:
    print(f"\n>>> Evaluating: {c['name']} ...")
    rep = evaluate_candidate_challenger(
        candidate_cls=c["class"],
        challenger_name=c["name"],
        problem_addressed=c["problem"],
        selected_cycles=benchmark_cycles,
        min_trade_freq_per_hour=1.0,
        min_win_rate_pct=70.0,
    )
    reports.append(rep)
    status_str = "PASSED [QUALIFIED]" if rep.passed_all_gates else "REJECTED [NOT QUALIFIED]"
    print(f"Status: {status_str} | Total Trades: {rep.total_trades} | Win Rate: {rep.overall_win_rate_pct}% | Net PnL: {rep.total_net_pnl_bps:+.2f} bps | Avg Tr/Hr: {rep.avg_trades_per_hour}")
    if rep.disqualification_reasons:
        for r in rep.disqualification_reasons:
            print(f"  - Gate Failure: {r}")
    print("  Cycle Breakdown:")
    for cb in rep.cycle_breakdown:
        print(f"    * {cb.cycle_name}: {cb.trades_count} tr | {cb.wins_count}W/{cb.losses_count}L ({cb.win_rate_pct}%) | Net: {cb.net_pnl_bps:+.2f} bps | PF: {cb.profit_factor}")

# Save full qualification report to scratch
qualification_json = [
    {
        "name": r.challenger_name,
        "problem_addressed": r.problem_addressed,
        "passed": r.passed_all_gates,
        "total_trades": r.total_trades,
        "win_rate_pct": r.overall_win_rate_pct,
        "total_net_pnl_bps": r.total_net_pnl_bps,
        "avg_trades_per_hour": r.avg_trades_per_hour,
        "disqualification_reasons": r.disqualification_reasons,
        "cycle_results": [
            {
                "cycle": cr.cycle_name,
                "trades": cr.trades_count,
                "wins": cr.wins_count,
                "losses": cr.losses_count,
                "win_rate": cr.win_rate_pct,
                "net_pnl": cr.net_pnl_bps
            }
            for cr in r.cycle_breakdown
        ]
    }
    for r in reports
]

with open("scratch/challenger_historical_qualification_report.json", "w", encoding="utf-8") as f:
    json.dump(qualification_json, f, indent=2)

print("\n" + "=" * 80)
print("HISTORICAL QUALIFICATION COMPLETE - SAVED TO scratch/challenger_historical_qualification_report.json")
print("=" * 80)
