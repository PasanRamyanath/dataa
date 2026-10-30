import sys, os, json
sys.path.insert(0, os.path.abspath('feature engineering scripts'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from testing.challenger_historical_qualification_harness import (
    HistoricalLedgerReplayer,
    evaluate_candidate_challenger,
)
from execution.champion_challenger_framework import (
    Cycle35_ReversionPureScalper,
    Cycle35_TightRangeReversionScalper,
    Cycle60_ExhaustionAwareScalper,
    Cycle60_ActiveFrequencyScalper,
    Cycle60_SteppedRatchetScalper,
    Cycle60_DynamicAsymmetricScalper,
)

benchmark_cycles = [
    "archive_cycle12",
    "archive_cycle17",
    "archive_cycle35",
    "archive_cycle48",
    "archive_cycle57",
    "archive_cycle58",
    "archive_cycle59",
]

competitor_lineup = [
    {
        "slot": "Champion",
        "name": "Reigning Champion: Cycle 35 Reversion Pure Scalper",
        "class": Cycle35_ReversionPureScalper,
        "problem": "Foundational Conformal Certainty Baseline (+111.63 bps Peak Cumulative Net)",
    },
    {
        "slot": "Challenger 1",
        "name": "Challenger 1: Cycle 59 Winner ($6 Range Envelope Scalper)",
        "class": Cycle35_TightRangeReversionScalper,
        "problem": "Solves range midpoint whipsaw by entering within $6 of 15m extremes (+14.58b maker win in Cycle 59)",
    },
    {
        "slot": "Challenger 2",
        "name": "Challenger 2: Exhaustion-Aware $6 Envelope Scalper",
        "class": Cycle60_ExhaustionAwareScalper,
        "problem": "Solves tail-chasing losses (Cycle 59 Trade 3) by vetoing entries in bottom/top 20% of extended range",
    },
    {
        "slot": "Challenger 3",
        "name": "Challenger 3: Active Frequency $5 Envelope Scalper",
        "class": Cycle60_ActiveFrequencyScalper,
        "problem": "Solves trade starvation / overly conservative 0-trade sessions by lowering edge hurdle to 3.5 bps",
    },
    {
        "slot": "Challenger 4",
        "name": "Challenger 4: Stepped Ratchet $6 Envelope Scalper",
        "class": Cycle60_SteppedRatchetScalper,
        "problem": "Solves profit decay by locking +1.50b net at +4.5b and +3.50b net at +7.5b MFE",
    },
    {
        "slot": "Challenger 5",
        "name": "Challenger 5: Dynamic Asymmetric Scalper",
        "class": Cycle60_DynamicAsymmetricScalper,
        "problem": "Combines $6 envelope, exhaustion guard, and fast trailing locks to maximize trade quality",
    },
]

print("=" * 90)
print("COMPREHENSIVE MULTI-CYCLE HISTORICAL QUALIFICATION AUDIT FOR TOURNAMENT LINEUP")
print("=" * 90)

audit_results = []
for comp in competitor_lineup:
    print(f"\n--- AUDITING: [{comp['slot']}] {comp['name']} ---")
    print(f"Problem Addressed: {comp['problem']}")
    
    tot_trades = 0
    tot_wins = 0
    tot_losses = 0
    tot_pnl = 0.0
    c_records = []
    
    for c_dir in benchmark_cycles:
        path = f"scratch/{c_dir}/live_hierarchical_session_1h_ledger.jsonl"
        if not os.path.exists(path):
            continue
        rep = HistoricalLedgerReplayer(path)
        eng = comp["class"](symbol="BTCUSDT")
        trades = rep.replay_into_engine(eng)
        wins = [t for t in trades if t.net_return_bps > 0]
        losses = [t for t in trades if t.net_return_bps <= 0]
        pnl = sum(t.net_return_bps for t in trades)
        wr = (len(wins) / len(trades) * 100.0) if trades else 0.0
        
        tot_trades += len(trades)
        tot_wins += len(wins)
        tot_losses += len(losses)
        tot_pnl += pnl
        
        c_records.append({
            "cycle": c_dir.replace("archive_", ""),
            "trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(wr, 1),
            "net_pnl": round(pnl, 2),
        })
        print(f"  * {c_dir:16s}: {len(trades):2d} trades | {len(wins)}W/{len(losses)}L ({wr:5.1f}%) | Net: {pnl:+6.2f} bps")
        
    avg_trades = round(tot_trades / len(benchmark_cycles), 2)
    overall_wr = round((tot_wins / tot_trades * 100.0) if tot_trades > 0 else 0.0, 1)
    
    audit_results.append({
        "slot": comp["slot"],
        "name": comp["name"],
        "problem_addressed": comp["problem"],
        "total_trades": tot_trades,
        "avg_trades_per_hour": avg_trades,
        "overall_win_rate": overall_wr,
        "total_net_pnl": round(tot_pnl, 2),
        "cycle_breakdown": c_records
    })
    print(f"Summary for {comp['slot']}: {tot_trades} trades ({avg_trades} tr/hr) | Win Rate: {overall_wr}% | Net PnL: {tot_pnl:+.2f} bps")

with open("scratch/full_tournament_qualification_audit.json", "w", encoding="utf-8") as f:
    json.dump(audit_results, f, indent=2)

print("\n" + "=" * 90)
print("AUDIT COMPLETE - RESULTS SAVED TO scratch/full_tournament_qualification_audit.json")
print("=" * 90)
