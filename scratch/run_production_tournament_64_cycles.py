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
    Cycle65_UnifiedPeakReversionEngine,
    Cycle65_PureMakerMeanReversionScalper,
    ChampionChallengerTournamentManager
)

base_dir = "D:/PrivateProjects/BTC Futures/scratch"
archives = sorted([d for d in os.listdir(base_dir) if (d.startswith("archive_cycle") or d.startswith("archive_v3")) and os.path.isdir(os.path.join(base_dir, d))])

def evaluate_engine_across_all_archives(engine_cls, engine_title):
    all_trades = []
    active_sessions = 0
    wr100_sessions = 0
    profitable_sessions = 0
    exit_reasons = Counter()

    for arch in archives:
        ledgers = glob.glob(os.path.join(base_dir, arch, "*ledger*.jsonl"))
        if not ledgers:
            continue
        with open(ledgers[0], "r", encoding="utf-8") as f:
            bars = [json.loads(l) for l in f if l.strip().startswith("{")]

        engine = engine_cls(symbol="BTCUSDT")

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
            sess_net = sum(t.net_return_bps for t in engine.closed_trades)
            if w == len(engine.closed_trades):
                wr100_sessions += 1
            if sess_net > 0:
                profitable_sessions += 1

            for t in engine.closed_trades:
                exit_reasons[t.exit_reason] += 1
                all_trades.append({
                    "cycle": arch,
                    "side": t.side,
                    "gross": t.gross_return_bps,
                    "net": t.net_return_bps,
                    "fee": t.total_fee_bps,
                    "dur": t.holding_duration_sec,
                    "reason": t.exit_reason
                })

    n = len(all_trades)
    w = sum(1 for t in all_trades if t["net"] > 0)
    l = n - w
    wr = (w / n * 100.0) if n > 0 else 0.0
    tot_pnl = sum(t["net"] for t in all_trades)
    avg_pnl = (tot_pnl / n) if n > 0 else 0.0

    print("=" * 100)
    print(f"OFFICIAL PRODUCTION AUDIT: {engine_title}")
    print("=" * 100)
    print(f"Total Closed Trades: {n} across {active_sessions} active sessions ({n/max(1, active_sessions):.2f} trades/session)")
    print(f"Wins: {w} | Losses: {l} | Win Rate: {wr:.2f}%")
    print(f"Total Net PnL: {tot_pnl:+.2f} bps NET REALIZED PROFIT")
    print(f"Average Profit per Trade: {avg_pnl:+.2f} bps/trade")
    print(f"100% Win Rate Sessions: {wr100_sessions} / {active_sessions} ({wr100_sessions/max(1, active_sessions)*100:.1f}%)")
    print(f"Profitable Sessions: {profitable_sessions} / {active_sessions} ({profitable_sessions/max(1, active_sessions)*100:.1f}%)")
    print("\nExit Mechanism Distribution:")
    for r, cnt in exit_reasons.most_common():
        sub_pnl = sum(t["net"] for t in all_trades if t["reason"] == r)
        print(f"  {r:<35}: {cnt:>3} trades | Net PnL: {sub_pnl:>+8.2f} bps | Avg: {sub_pnl/cnt:>+5.2f} bps")
    print("=" * 100 + "\n")
    return {
        "trades": n, "wins": w, "losses": l, "win_rate": wr,
        "net_pnl": tot_pnl, "avg_pnl": avg_pnl, "active_sessions": active_sessions,
        "wr100_sessions": wr100_sessions
    }

if __name__ == "__main__":
    c1_res = evaluate_engine_across_all_archives(
        Cycle65_UnifiedPeakReversionEngine,
        "Challenger 1: Cycle65_UnifiedPeakReversionEngine (Two-Way Integrated Peak Engine)"
    )
    c5_res = evaluate_engine_across_all_archives(
        Cycle65_PureMakerMeanReversionScalper,
        "Challenger 5: Cycle65_PureMakerMeanReversionScalper (Long-Only Peak Engine)"
    )
