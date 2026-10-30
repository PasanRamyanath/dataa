import os, glob, json, sys, copy, types
from collections import Counter
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append('feature engineering scripts')
from scratch.test_quad_mode_multi_tranche_replay import Cycle66_QuadModeMultiTranchePeakEngine

base_dir = "scratch"
archives = sorted([d for d in os.listdir(base_dir) if (d.startswith("archive_cycle") or d.startswith("archive_v3")) and os.path.isdir(os.path.join(base_dir, d))])

def test_cooldown(cd_sec):
    all_closed_trades = []
    unclosed_trades = []

    for arch in archives:
        ledgers = glob.glob(os.path.join(base_dir, arch, "*ledger*.jsonl"))
        if not ledgers: continue
        with open(ledgers[0], "r", encoding="utf-8") as f:
            bars = [json.loads(line) for line in f if line.strip().startswith("{")]

        engine = Cycle66_QuadModeMultiTranchePeakEngine(symbol="BTCUSDT")

        for idx, bar in enumerate(bars):
            mid = bar.get("bbo_mid", bar.get("mid", 0.0))
            bid = bar.get("bbo_bid", mid - 0.05)
            ask = bar.get("bbo_ask", mid + 0.05)
            ts_ms = bar.get("ts_ms", idx * 10000)

            decision = types.SimpleNamespace(
                ts_ms=ts_ms,
                primary_direction=bar.get("primary_direction", 1),
                expected_net_edge_bps=bar.get("expected_net_edge_bps", 0.0),
                tactical_probs=bar.get("tactical_probs", [0.33, 0.33, 0.33]),
                meso_probs=bar.get("meso_probs", [0.33, 0.33, 0.33]),
                macro_probs=bar.get("macro_probs", [0.33, 0.33, 0.33]),
                micro_probs=bar.get("micro_probs", [0.33, 0.33, 0.33]),
                kelly_position_fraction=0.15,
                execution_mode="PASSIVE_MAKER_SNIPER",
                target_holding_seconds=900,
                take_profit_bps=35.0,
                stop_loss_bps=45.0,
                micro_sniper_trigger_prob=0.85
            )

            # Check cooldown before decision
            if engine.last_exit_ts_ms > 0 and (ts_ms - engine.last_exit_ts_ms) / 1000.0 < cd_sec:
                engine.on_market_micro_tick(ts_ms=ts_ms, bbo_bid=bid, bbo_ask=ask, bbo_bid_qty=10, bbo_ask_qty=10, bar_high=ask+0.05, bar_low=bid-0.05, current_decision=decision)
                continue

            engine.on_market_micro_tick(ts_ms=ts_ms, bbo_bid=bid, bbo_ask=ask, bbo_bid_qty=10, bbo_ask_qty=10, bar_high=ask+0.05, bar_low=bid-0.05, current_decision=decision)
            engine.on_hierarchical_decision(decision=decision, bbo_bid=bid, bbo_ask=ask, bbo_bid_qty=10, bbo_ask_qty=10, equity_usdt=10000.0, intra_bar_open=mid)
            if engine.state == "RESTING_ENTRY_QUEUE":
                engine.on_market_micro_tick(ts_ms=ts_ms+1000, bbo_bid=bid, bbo_ask=ask, bbo_bid_qty=10, bbo_ask_qty=10, bar_high=ask+0.05, bar_low=bid-0.05, current_decision=decision)

        if engine.active_position is not None:
            last_mid = bars[-1].get("bbo_mid", bars[-1].get("mid", 0.0))
            pos = engine.active_position
            sign = 1.0 if pos.side == "BUY" else -1.0
            u_bps = sign * ((last_mid - engine.pos_avg_entry_price) / engine.pos_avg_entry_price) * 1e4
            net_bps = u_bps - 4.0
            unclosed_trades.append({"cycle": arch, "net": net_bps})

        for t in engine.closed_trades:
            all_closed_trades.append(t.net_return_bps)

    n = len(all_closed_trades)
    w = sum(1 for x in all_closed_trades if x > 0)
    l = n - w
    tot_pnl = sum(all_closed_trades)
    print(f"CD={cd_sec:3d}s: Trades={n:>3d} | W={w:>3d} | L={l:>2d} | WR={(w/max(1,n))*100:>5.2f}% | Net={tot_pnl:>+8.2f} bps | Unclosed={len(unclosed_trades)}")

for cd in [60, 90, 120, 150, 180, 240, 300]:
    test_cooldown(cd)
