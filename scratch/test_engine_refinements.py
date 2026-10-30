import os
import glob
import json
import io
import sys
import types
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.append('feature engineering scripts')

from execution.champion_challenger_framework import Cycle66_QuadModeMultiTranchePeakEngine

base_dir = "D:/PrivateProjects/BTC Futures/scratch"
archives = sorted([d for d in os.listdir(base_dir) if (d.startswith("archive_cycle") or d.startswith("archive_v3")) and os.path.isdir(os.path.join(base_dir, d))])

def test_engine_variant(scale_meso_filter=True, min_scale_rem_bars=25, session_harvest_gross=2.50):
    all_closed_trades = []
    loss_trades = []
    
    for arch in archives:
        ledgers = glob.glob(os.path.join(base_dir, arch, "*ledger*.jsonl"))
        if not ledgers:
            continue
        with open(ledgers[0], "r", encoding="utf-8") as f:
            bars = [json.loads(line) for line in f if line.strip().startswith("{")]

        engine = Cycle66_QuadModeMultiTranchePeakEngine(symbol="BTCUSDT")
        total_session_bars = len(bars)

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

            actual_rem_bars = total_session_bars - 1 - idx

            # Micro tick with actual remaining bars
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

            # Hierarchical decision
            engine.on_hierarchical_decision(
                decision=decision,
                bbo_bid=bid,
                bbo_ask=ask,
                bbo_bid_qty=bid_qty,
                bbo_ask_qty=ask_qty,
                equity_usdt=10000.0,
                intra_bar_open=mid
            )

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

        if engine.active_position is not None:
            last_mid = bars[-1].get("bbo_mid", bars[-1].get("mid", 0.0))
            pos = engine.active_position
            sign = 1.0 if pos.side == "BUY" else -1.0
            avg_ep = getattr(engine, "pos_avg_entry_price", pos.entry_price) or pos.entry_price
            u_bps = sign * ((last_mid - avg_ep) / avg_ep) * 1e4
            net_m2m = u_bps - 4.0
            engine._close_position(exit_price=last_mid, exit_ts_ms=bars[-1].get("ts_ms", 3600000), exit_fee_bps=2.0, reason="SESSION_END_MARK_TO_MARKET")

        for t in engine.closed_trades:
            rec = {
                "cycle": arch,
                "side": t.side,
                "gross": t.gross_return_bps,
                "net": t.net_return_bps,
                "reason": t.exit_reason
            }
            all_closed_trades.append(rec)
            if t.net_return_bps <= 0:
                loss_trades.append(rec)

    n = len(all_closed_trades)
    w = sum(1 for t in all_closed_trades if t["net"] > 0)
    l = n - w
    wr = (w / n * 100.0) if n > 0 else 0.0
    tot_pnl = sum(t["net"] for t in all_closed_trades)
    print(f"Results: Total={n} | Wins={w} | Losses={l} | WR={wr:.2f}% | Net PnL={tot_pnl:+.2f} bps")
    print(f"Losses: {[t['cycle'] + '_' + t['side'] + '(' + str(round(t['net'], 1)) + 'b, ' + t['reason'] + ')' for t in loss_trades]}")

test_engine_variant()
