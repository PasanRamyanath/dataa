import os
import glob
import json
import io
import sys
import types
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')
sys.path.append('feature engineering scripts')

from execution.champion_challenger_framework import Cycle66_QuadModeMultiTranchePeakEngine
from exact_quoted_architecture_replay import ExactQuotedCycle66Engine

base_dir = "D:/PrivateProjects/BTC Futures/scratch"
archives = sorted([d for d in os.listdir(base_dir) if (d.startswith("archive_cycle") or d.startswith("archive_v3")) and os.path.isdir(os.path.join(base_dir, d))])

session_files = []
for arch in archives:
    ledgers = glob.glob(os.path.join(base_dir, arch, "*ledger*.jsonl"))
    if ledgers:
        session_files.append((arch, ledgers[0]))
live_1h = os.path.join(base_dir, "live_hierarchical_session_1h_ledger.jsonl")
if os.path.exists(live_1h):
    session_files.append(("cycle65_live_1h", live_1h))

def evaluate_engine(engine_cls, label):
    all_trades = []
    unclosed = []
    active_sess = 0
    tranche_fills = Counter()

    for sess_name, ledger_path in session_files:
        with open(ledger_path, "r", encoding="utf-8") as f:
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
                ts_ms=ts_ms, primary_direction=p_dir, expected_net_edge_bps=edge,
                tactical_probs=tact, meso_probs=meso, macro_probs=macro, micro_probs=micro,
                kelly_position_fraction=0.15, execution_mode="PASSIVE_MAKER_SNIPER",
                target_holding_seconds=900, take_profit_bps=35.0, stop_loss_bps=45.0,
                micro_sniper_trigger_prob=0.85
            )

            prev_pos = engine.active_position
            prev_t = engine.pos_tranches if prev_pos else 0

            engine.on_market_micro_tick(ts_ms=ts_ms, bbo_bid=bid, bbo_ask=ask, bbo_bid_qty=bid_qty, bbo_ask_qty=ask_qty, current_decision=decision)
            engine.on_hierarchical_decision(decision=decision, bbo_bid=bid, bbo_ask=ask, bbo_bid_qty=bid_qty, bbo_ask_qty=ask_qty, equity_usdt=10000.0, intra_bar_open=mid)

            if engine.state == "RESTING_ENTRY_QUEUE":
                engine.on_market_micro_tick(ts_ms=ts_ms+1000, bbo_bid=bid, bbo_ask=ask, bbo_bid_qty=bid_qty, bbo_ask_qty=ask_qty, current_decision=decision)

            if engine.active_position is not None:
                if prev_pos is None:
                    tranche_fills[1] += 1
                elif engine.pos_tranches > prev_t:
                    for t_idx in range(prev_t + 1, engine.pos_tranches + 1):
                        tranche_fills[t_idx] += 1

        if engine.active_position is not None:
            last_mid = bars[-1].get("bbo_mid", bars[-1].get("mid", 0.0))
            pos = engine.active_position
            sign = 1.0 if pos.side == "BUY" else -1.0
            avg_ep = getattr(engine, "pos_avg_entry_price", pos.entry_price) or pos.entry_price
            u_bps = sign * ((last_mid - avg_ep) / avg_ep) * 1e4
            net_m2m = u_bps - 4.0
            unclosed.append({"cycle": sess_name, "side": pos.side, "net": net_m2m, "gross": u_bps, "tranches": engine.pos_tranches})
            engine._close_position(exit_price=last_mid, exit_ts_ms=bars[-1].get("ts_ms", 3600000), exit_fee_bps=2.0, reason="SESSION_END_MARK_TO_MARKET")

        if engine.closed_trades:
            active_sess += 1
            for t in engine.closed_trades:
                all_trades.append({
                    "cycle": sess_name, "side": t.side, "net": t.net_return_bps, "gross": t.gross_return_bps,
                    "reason": t.exit_reason, "weight": getattr(engine, "pos_weight", 1.0)
                })

    n = len(all_trades)
    w = sum(1 for t in all_trades if t["net"] > 0)
    l = n - w
    wr = (w / n * 100.0) if n > 0 else 0.0
    tot_pnl = sum(t["net"] for t in all_trades)
    w_pnl = sum(t["net"] * t.get("weight", 1.0) for t in all_trades)
    avg_pnl = tot_pnl / n if n > 0 else 0.0
    gross_w = sum(t["net"] for t in all_trades if t["net"] > 0)
    gross_l = abs(sum(t["net"] for t in all_trades if t["net"] <= 0))
    pf = (gross_w / gross_l) if gross_l > 0 else 999.0
    hard_stops = sum(1 for t in all_trades if t["reason"] == "HARD_STOP_LOSS_TAKER")

    print(f"\n{'='*90}")
    print(f"EVALUATION: {label}")
    print(f"{'='*90}")
    print(f"Total Sessions: {len(session_files)} (Active: {active_sess})")
    print(f"Total Completed Positions: {n} ({n/active_sess:.2f}/sess)")
    print(f"Total Individual Tranche Fills: {sum(tranche_fills.values())} (T1={tranche_fills[1]}, T2={tranche_fills[2]}, T3={tranche_fills[3]}, T4={tranche_fills[4]})")
    print(f"Wins: {w} | Losses: {l} | Win Rate: {wr:.2f}% | Profit Factor: {pf:.2f}")
    print(f"Hard Stops: {hard_stops} ({(hard_stops/n*100):.2f}%)")
    print(f"Unweighted Net Realized Profit: {tot_pnl:+.2f} bps | Avg: {avg_pnl:+.2f} bps/trade")
    print(f"Tranche-Weighted Net Profit   : {w_pnl:+.2f} bps")
    print(f"Unclosed Positions at Session End: {len(unclosed)}")
    for uc in unclosed:
        print(f"  * {uc['cycle']} {uc['side']} (Tranches={uc['tranches']}): Gross={uc['gross']:+.2f}b, Net={uc['net']:+.2f}b")

evaluate_engine(ExactQuotedCycle66Engine, "1. Exact Prompt Architecture (Raw m3 dm >= 0.03)")
evaluate_engine(Cycle66_QuadModeMultiTranchePeakEngine, "2. Production Supreme Champion Architecture (Enhanced m3 dm >= 0.08 if not p_ok)")
