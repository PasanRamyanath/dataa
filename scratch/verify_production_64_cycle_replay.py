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
    Cycle66_QuadModeMultiTranchePeakEngine,
    Cycle65_PeakBidirectionalFuturesScalper,
    Cycle65_FlawlessBidirectionalScalper,
    Cycle65_UnifiedPeakReversionEngine,
    Cycle65_PureMakerMeanReversionScalper,
    Cycle35_ReversionPureScalper,
    ChampionChallengerTournamentManager
)

base_dir = "D:/PrivateProjects/BTC Futures/scratch"
archives = sorted([d for d in os.listdir(base_dir) if (d.startswith("archive_cycle") or d.startswith("archive_v3")) and os.path.isdir(os.path.join(base_dir, d))])

def run_production_engine_replay(engine_cls, name):
    all_closed_trades = []
    long_trades = []
    short_trades = []
    unclosed_trades = []
    active_sessions = 0
    wr100_sessions = 0
    profitable_sessions = 0
    exit_reasons = Counter()

    for arch in archives:
        ledgers = glob.glob(os.path.join(base_dir, arch, "*ledger*.jsonl"))
        if not ledgers:
            continue
        with open(ledgers[0], "r", encoding="utf-8") as f:
            bars = [json.loads(line) for line in f if line.strip().startswith("{")]

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

            # 1. Market micro tick
            engine.on_market_micro_tick(
                ts_ms=ts_ms,
                bbo_bid=bid,
                bbo_ask=ask,
                bbo_bid_qty=bid_qty,
                bbo_ask_qty=ask_qty,
                bar_high=ask + 0.05,
                bar_low=bid - 0.05,
                current_decision=decision
            )

            # 2. Hierarchical decision
            engine.on_hierarchical_decision(
                decision=decision,
                bbo_bid=bid,
                bbo_ask=ask,
                bbo_bid_qty=bid_qty,
                bbo_ask_qty=ask_qty,
                equity_usdt=10000.0,
                intra_bar_open=mid
            )

            # 3. Post-decision fill evaluation if resting entry placed
            if engine.state == "RESTING_ENTRY_QUEUE":
                engine.on_market_micro_tick(
                    ts_ms=ts_ms + 1000,
                    bbo_bid=bid,
                    bbo_ask=ask,
                    bbo_bid_qty=bid_qty,
                    bbo_ask_qty=ask_qty,
                    bar_high=ask + 0.05,
                    bar_low=bid - 0.05,
                    current_decision=decision
                )

        # Honest Session-End Mark-to-Market check
        if engine.active_position is not None:
            last_mid = bars[-1].get("bbo_mid", bars[-1].get("mid", 0.0))
            pos = engine.active_position
            sign = 1.0 if pos.side == "BUY" else -1.0
            avg_ep = getattr(engine, "pos_avg_entry_price", pos.entry_price) or pos.entry_price
            u_bps = sign * ((last_mid - avg_ep) / avg_ep) * 1e4
            net_m2m = u_bps - 4.0  # 4.00 bps round-trip maker fee
            unclosed_trades.append({
                "cycle": arch, "side": pos.side, "net": net_m2m, "gross": u_bps, "reason": "SESSION_END_MARK_TO_MARKET"
            })
            engine._close_position(exit_price=last_mid, exit_ts_ms=bars[-1].get("ts_ms", 3600000), exit_fee_bps=2.0, reason="SESSION_END_MARK_TO_MARKET")

        session_trades = engine.closed_trades
        if session_trades:
            active_sessions += 1
            wins = sum(1 for t in session_trades if t.net_return_bps > 0)
            losses = sum(1 for t in session_trades if t.net_return_bps <= 0)
            sess_net = sum(t.net_return_bps for t in session_trades)
            if wins == len(session_trades):
                wr100_sessions += 1
            if sess_net > 0:
                profitable_sessions += 1

            for t in session_trades:
                exit_reasons[t.exit_reason] += 1
                rec = {
                    "cycle": arch,
                    "side": t.side,
                    "gross": t.gross_return_bps,
                    "net": t.net_return_bps,
                    "fee": t.total_fee_bps,
                    "dur": t.holding_duration_sec,
                    "reason": t.exit_reason,
                    "weight": getattr(engine, "pos_weight", 1.0)
                }
                all_closed_trades.append(rec)
                if t.side == "BUY":
                    long_trades.append(rec)
                else:
                    short_trades.append(rec)

    n = len(all_closed_trades)
    w = sum(1 for t in all_closed_trades if t["net"] > 0)
    l = n - w
    wr = (w / n * 100.0) if n > 0 else 0.0
    tot_pnl = sum(t["net"] for t in all_closed_trades)
    tot_gross = sum(t["gross"] for t in all_closed_trades)
    tot_fee = sum(t["fee"] for t in all_closed_trades)
    avg_pnl = (tot_pnl / n) if n > 0 else 0.0

    n_l = len(long_trades)
    w_l = sum(1 for t in long_trades if t["net"] > 0)
    wr_l = (w_l / n_l * 100.0) if n_l > 0 else 0.0
    pnl_l = sum(t["net"] for t in long_trades)

    n_s = len(short_trades)
    w_s = sum(1 for t in short_trades if t["net"] > 0)
    wr_s = (w_s / n_s * 100.0) if n_s > 0 else 0.0
    pnl_s = sum(t["net"] for t in short_trades)

    gw = sum(t["net"] for t in all_closed_trades if t["net"] > 0)
    gl = abs(sum(t["net"] for t in all_closed_trades if t["net"] <= 0))
    pf = (gw / gl) if gl > 0 else 999.0

    loss_trades = [t for t in all_closed_trades if t["net"] <= 0]
    loss_str = ", ".join([f"{x['cycle']}_{x['side']} ({x['net']:.1f}b, {x['reason']})" for x in loss_trades]) or "None (0 Losses)"

    print("=" * 105)
    print(f"VERIFICATION AUDIT (HONEST M2M): {name}")
    print("=" * 105)
    print(f"Total Completed Positions: {n} across {active_sessions} active sessions ({n/max(1, active_sessions):.2f} trades/session)")
    print(f"Wins: {w} | Losses: {l} | Win Rate: {wr:.2f}% | Profit Factor: {pf:.2f}")
    print(f"Gross PnL: {tot_gross:+.2f} bps | Total Fees: {tot_fee:.2f} bps | Net PnL: {tot_pnl:+.2f} bps NET REALIZED PROFIT")
    print(f"Average Profit per Trade: {avg_pnl:+.2f} bps/trade")
    print(f"  * LONGS : {n_l} trades | {w_l}W / {n_l - w_l}L | Win Rate: {wr_l:.2f}% | Net PnL: {pnl_l:+.2f} bps")
    print(f"  * SHORTS: {n_s} trades | {w_s}W / {n_s - w_s}L | Win Rate: {wr_s:.2f}% | Net PnL: {pnl_s:+.2f} bps")
    print(f"100% Win Rate Sessions: {wr100_sessions} / {active_sessions} ({wr100_sessions/max(1, active_sessions)*100:.1f}%)")
    print(f"Profitable Sessions: {profitable_sessions} / {active_sessions} ({profitable_sessions/max(1, active_sessions)*100:.1f}%)")
    print(f"Unclosed Stranded Positions at Session End: {len(unclosed_trades)}")
    print(f"Loss Trades: {loss_str}")
    print("\nExit Mechanism Distribution:")
    for r, cnt in exit_reasons.most_common():
        sub_pnl = sum(t["net"] for t in all_closed_trades if t["reason"] == r)
        print(f"  {r:<35}: {cnt:>3} trades | Net PnL: {sub_pnl:>+8.2f} bps | Avg: {sub_pnl/cnt:>+5.2f} bps")
    print("=" * 105 + "\n")
    return {
        "name": name,
        "trades": n, "wins": w, "losses": l, "win_rate": wr,
        "net_pnl": tot_pnl, "profit_factor": pf, "avg_pnl": avg_pnl,
        "long_trades": n_l, "long_wr": wr_l, "long_pnl": pnl_l,
        "short_trades": n_s, "short_wr": wr_s, "short_pnl": pnl_s,
        "active_sessions": active_sessions, "wr100_sessions": wr100_sessions,
        "stranded_positions": len(unclosed_trades)
    }

if __name__ == "__main__":
    results = []
    # 1. Reigning Supreme Champion: Cycle66_QuadModeMultiTranchePeakEngine
    results.append(run_production_engine_replay(
        Cycle66_QuadModeMultiTranchePeakEngine,
        "Supreme Champion: Cycle 66 Quad-Mode Multi-Tranche Peak Engine"
    ))
    # 2. Challenger 1: Cycle65_PeakBidirectionalFuturesScalper
    results.append(run_production_engine_replay(
        Cycle65_PeakBidirectionalFuturesScalper,
        "Challenger 1: Cycle 65 Peak Bidirectional Scalper"
    ))
    # 3. Challenger 2: Cycle65_FlawlessBidirectionalScalper
    results.append(run_production_engine_replay(
        Cycle65_FlawlessBidirectionalScalper,
        "Challenger 2: Cycle 65 Flawless Bidirectional Scalper"
    ))
    # 4. Challenger 3: Cycle65_UnifiedPeakReversionEngine
    results.append(run_production_engine_replay(
        Cycle65_UnifiedPeakReversionEngine,
        "Challenger 3: Cycle 65 Unified Peak Reversion Engine"
    ))
    # 5. Challenger 4: Cycle65_PureMakerMeanReversionScalper
    results.append(run_production_engine_replay(
        Cycle65_PureMakerMeanReversionScalper,
        "Challenger 4: Pure Maker Long-Only Mean Reversion Scalper"
    ))
    # 6. Challenger 5: Cycle35_ReversionPureScalper
    results.append(run_production_engine_replay(
        Cycle35_ReversionPureScalper,
        "Challenger 5: Cycle 35 Reversion Pure Scalper"
    ))

    print("\n" + "=" * 115)
    print(f"{'RANK':<5} | {'SYSTEM NAME':<45} | {'TRADES':<7} | {'WIN RATE':<9} | {'NET PNL':<12} | {'PF':<6} | {'AVG BPS':<8}")
    print("=" * 115)
    results.sort(key=lambda x: (x["net_pnl"], x["win_rate"]), reverse=True)
    for i, r in enumerate(results, 1):
        print(f"{i:<5} | {r['name']:<45} | {r['trades']:<7} | {r['win_rate']:>6.2f}%  | {r['net_pnl']:>+8.2f} bps | {r['profit_factor']:>5.2f} | {r['avg_pnl']:>+6.2f}b")
    print("=" * 115)
