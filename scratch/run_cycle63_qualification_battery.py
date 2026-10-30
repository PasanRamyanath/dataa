import sys, os, json
sys.path.insert(0, os.path.abspath('feature engineering scripts'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from testing.challenger_historical_qualification_harness import evaluate_candidate_challenger
from execution.champion_challenger_framework import (
    Cycle35_ReversionPureScalper,
    Cycle63_PassiveMakerUnwindScalper,
    Cycle63_PostInvalidationSpacingScalper,
    Cycle63_SteppedRatchetMidpointScalper,
    Cycle63_HighEdgeExpansionScalper,
    Cycle63_PureMicroMomentumScalper,
)

print("=" * 90)
print("RUNNING CYCLE 63 CHALLENGER QUALIFICATION BATTERY ACROSS BENCHMARK SUITE")
print("=" * 90)

candidates = [
    {
        "class": Cycle35_ReversionPureScalper,
        "name": "👑 Reigning Champion (Baseline)",
        "problem": "Baseline benchmark (+106.03 bps Defended Peak Cumulative Net)"
    },
    {
        "class": Cycle63_PassiveMakerUnwindScalper,
        "name": "⚔️ Challenger 1 Candidate (Passive Maker Unwind Scalper)",
        "problem": "Eliminates 5.0 bps Taker stop fee penalty on invalidation cuts, saving 3.0 bps per loss"
    },
    {
        "class": Cycle63_PostInvalidationSpacingScalper,
        "name": "⚔️ Challenger 2 Candidate (Post-Invalidation Spacing Scalper)",
        "problem": "Eliminates consecutive stop-outs in same band (Cycle 62 Trade 3 whipsaw)"
    },
    {
        "class": Cycle63_SteppedRatchetMidpointScalper,
        "name": "⚔️ Challenger 3 Candidate (Stepped Ratchet Midpoint Scalper)",
        "problem": "Eliminates profit decay on extended swings (+4.5 bps net locked at +8 bps MFE)"
    },
    {
        "class": Cycle63_HighEdgeExpansionScalper,
        "name": "⚔️ Challenger 4 Candidate (High Edge Expansion Scalper)",
        "problem": "Eliminates low-edge marginal chop entries (requires net edge >= 15.0 bps)"
    },
    {
        "class": Cycle63_PureMicroMomentumScalper,
        "name": "⚔️ Challenger 5 Candidate (Pure Micro-Momentum Filtered Scalper)",
        "problem": "Eliminates adverse sub-second micro sweeps (Cycle 60 100% Win Rate solution)"
    },
]

reports = []
for c in candidates:
    print(f"\nEvaluating {c['name']}...")
    rep = evaluate_candidate_challenger(
        candidate_cls=c["class"],
        challenger_name=c["name"],
        problem_addressed=c["problem"],
        min_trade_freq_per_hour=2.0,
        min_win_rate_pct=50.0,
    )
    reports.append(rep)
    status = "QUALIFIED FOR TOURNAMENT" if rep.passed_all_gates else "DISQUALIFIED [GATE RESTRICTION]"
    print(f"Status: {status}")
    print(f"Cycles: {rep.cycles_evaluated} | Trades: {rep.total_trades} | Wins: {rep.total_wins} | Losses: {rep.total_losses}")
    print(f"Win Rate: {rep.overall_win_rate_pct:.1f}% | Net PnL: {rep.total_net_pnl_bps:+.2f} bps | PF: {rep.overall_profit_factor:.2f} | Trades/Hr: {rep.avg_trades_per_hour:.2f}")
    if rep.disqualification_reasons:
        for r in rep.disqualification_reasons:
            print(f"  - Gate Note: {r}")

print("\n" + "=" * 90)
print("QUALIFICATION BATTERY COMPLETED")
print("=" * 90)
