import os
import glob
import json
import io
import sys
import types
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.append('feature engineering scripts')

from execution.champion_challenger_framework import (
    Cycle35_ReversionPureScalper,
    Cycle65_UnifiedPeakReversionEngine,
    Cycle65_DynamicSteppedRatchetScalper,
    Cycle65_ConformalHighPrecisionScalper,
    Cycle65_AsymmetricRunnerExpansionScalper,
    Cycle65_PureMakerMeanReversionScalper,
)

base_dir = "D:/PrivateProjects/BTC Futures/scratch"
archives = sorted([d for d in os.listdir(base_dir) if (d.startswith("archive_cycle") or d.startswith("archive_v3")) and os.path.isdir(os.path.join(base_dir, d))])

competitor_classes = {
    "Champion (Cycle 35)": (Cycle35_ReversionPureScalper, "Reigning Champion: Cycle 35 Reversion Pure Scalper"),
    "Challenger 1 (Peak Two-Way)": (Cycle65_UnifiedPeakReversionEngine, "Challenger 1: Cycle 65 Unified Peak Reversion Engine (92.7% WR)"),
    "Challenger 2 (Stepped Ratchet)": (Cycle65_DynamicSteppedRatchetScalper, "Challenger 2: Dynamic Stepped Ratchet Scalper"),
    "Challenger 3 (Conformal)": (Cycle65_ConformalHighPrecisionScalper, "Challenger 3: Conformal High Precision Scalper"),
    "Challenger 4 (Runner TP +45)": (Cycle65_AsymmetricRunnerExpansionScalper, "Challenger 4: Asymmetric Runner Expansion Scalper"),
    "Challenger 5 (Peak Long-Only)": (Cycle65_PureMakerMeanReversionScalper, "Challenger 5: Pure Maker Mean Reversion Scalper (96.1% WR)")
}

results = {}

for name, (cls, profile) in competitor_classes.items():
    all_trades = []
    active_sessions = 0
    wr100_sessions = 0

    for arch in archives:
        ledgers = glob.glob(os.path.join(base_dir, arch, "*ledger*.jsonl"))
        if not ledgers:
            continue
        with open(ledgers[0], "r", encoding="utf-8") as f:
            bars = [json.loads(l) for l in f if l.strip().startswith("{")]

        engine = cls(symbol="BTCUSDT")

        for idx, bar in enumerate(bars):
            mid = bar.get("bbo_mid", bar.get("mid", 0.0))
            bid = bar.get("bbo_bid", mid - 0.05)
            ask = bar.get("bbo_ask", mid + 0.05)
            bid_qty = bar.get("bbo_bid_qty", 10.0)
            ask_qty = bar.get("bbo_ask_qty", 10.0)
            ts_ms = bar.get("ts_ms", idx * 10000)

            tact = bar.get("tactical_probs", [0.33, 0.33, 0.33])
            meso = bar.get("meso_probs", [0.33, 0.33, 0.33])
            macro = bar.get("macro_probs", [0.33, 0.33, 0.33])
            micro = bar.get("micro_probs", [0.33, 0.33, 0.33])
            p_dir = bar.get("primary_direction", bar.get("direction", 1))
            edge = bar.get("expected_net_edge_bps", bar.get("edge", 0.0))

            decision = types.SimpleNamespace(
                ts_ms=ts_ms,
                primary_direction=p_dir,
                expected_net_edge_bps=edge,
                tactical_probs=tact,
                meso_probs=meso,
                macro_probs=macro,
                micro_probs=micro,
                kelly_position_fraction=0.15,
                execution_mode="PASSIVE_MAKER_SNIPER",
                target_holding_seconds=900,
                take_profit_bps=35.0,
                stop_loss_bps=45.0,
                micro_sniper_trigger_prob=0.85
            )

            engine.on_hierarchical_decision(decision, bid, ask, bid_qty, ask_qty, 10000.0, mid)
            engine.on_market_micro_tick(ts_ms, bid, ask, bid_qty, ask_qty, bar_high=ask + 0.05, bar_low=bid - 0.05, current_decision=decision)

        if engine.closed_trades:
            active_sessions += 1
            w = sum(1 for t in engine.closed_trades if t.net_return_bps > 0)
            if w == len(engine.closed_trades):
                wr100_sessions += 1
            for t in engine.closed_trades:
                all_trades.append(t)

    n = len(all_trades)
    w = sum(1 for t in all_trades if t.net_return_bps > 0)
    l = n - w
    wr = (w / n * 100.0) if n > 0 else 0.0
    tot_pnl = sum(t.net_return_bps for t in all_trades)
    avg_pnl = (tot_pnl / n) if n > 0 else 0.0
    gw = sum(t.net_return_bps for t in all_trades if t.net_return_bps > 0)
    gl = abs(sum(t.net_return_bps for t in all_trades if t.net_return_bps < 0))
    pf = (gw / max(1e-6, gl)) if gl > 0 else 99.0

    results[name] = {
        "profile": profile,
        "trades": n,
        "wins": w,
        "losses": l,
        "wr": wr,
        "net_pnl": tot_pnl,
        "avg_pnl": avg_pnl,
        "pf": pf,
        "active_sessions": active_sessions,
        "wr100_sessions": wr100_sessions,
    }

print("=" * 115)
print("🏆 OFFICIAL CYCLE 65 TOURNAMENT BENCHMARK ACROSS ALL 64 HISTORICAL CYCLES")
print("=" * 115)
sorted_results = sorted(results.items(), key=lambda x: (x[1]["net_pnl"], x[1]["wr"]), reverse=True)
print(f"{'Rank':<5} | {'Competitor':<30} | {'Trades':<6} | {'WR %':<7} | {'Net PnL (bps)':<14} | {'PF':<5} | {'100% WR Sess':<13} | {'Avg (bps)'}")
print("-" * 115)
for rank, (name, r) in enumerate(sorted_results, 1):
    print(f"#{rank:<4} | {name:<30} | {r['trades']:<6} | {r['wr']:>5.1f}% | {r['net_pnl']:>+12.2f} bps | {r['pf']:>4.2f} | {r['wr100_sessions']:>2}/{r['active_sessions']:<2} ({r['wr100_sessions']/max(1, r['active_sessions'])*100:4.1f}%) | {r['avg_pnl']:>+6.2f}")
print("=" * 115)
