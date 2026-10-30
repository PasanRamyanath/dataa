import sys, os, glob, json
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append('feature engineering scripts')
from scratch.test_cycle65_peak_bidirectional_class import Cycle65_PeakBidirectionalFuturesScalper

base_dir = "scratch"
archives = sorted([d for d in os.listdir(base_dir) if (d.startswith("archive_cycle") or d.startswith("archive_v3")) and os.path.isdir(os.path.join(base_dir, d))])

long_held_trades = []

for arch in archives:
    ledgers = glob.glob(os.path.join(base_dir, arch, "*ledger*.jsonl"))
    if not ledgers: continue
    with open(ledgers[0], "r", encoding="utf-8") as f:
        bars = [json.loads(l) for l in f if l.strip().startswith("{")]

    engine = Cycle65_PeakBidirectionalFuturesScalper()
    import types
    for idx, bar in enumerate(bars):
        mid = bar.get("bbo_mid", bar.get("mid", 0.0))
        bid = bar.get("bbo_bid", mid - 0.05)
        ask = bar.get("bbo_ask", mid + 0.05)
        ts_ms = bar.get("ts_ms", idx * 10000)
        decision = types.SimpleNamespace(
            ts_ms=ts_ms, primary_direction=bar.get("primary_direction", 1),
            expected_net_edge_bps=bar.get("expected_net_edge_bps", 0.0),
            tactical_probs=bar.get("tactical_probs", [0.33, 0.33, 0.33]),
            meso_probs=bar.get("meso_probs", [0.33, 0.33, 0.33]),
            macro_probs=bar.get("macro_probs", [0.33, 0.33, 0.33]),
            micro_probs=bar.get("micro_probs", [0.33, 0.33, 0.33]),
            kelly_position_fraction=0.15, execution_mode="PASSIVE_MAKER_SNIPER",
            target_holding_seconds=900, take_profit_bps=35.0, stop_loss_bps=45.0,
            micro_sniper_trigger_prob=0.85
        )
        engine.on_market_micro_tick(ts_ms=ts_ms, bbo_bid=bid, bbo_ask=ask, bbo_bid_qty=10, bbo_ask_qty=10, bar_high=ask+0.05, bar_low=bid-0.05, current_decision=decision)
        engine.on_hierarchical_decision(decision, bid, ask, 10, 10, 10000.0, mid)
        if engine.state == "RESTING_ENTRY_QUEUE":
            engine.on_market_micro_tick(ts_ms=ts_ms+1000, bbo_bid=bid, bbo_ask=ask, bbo_bid_qty=10, bbo_ask_qty=10, bar_high=ask+0.05, bar_low=bid-0.05, current_decision=decision)

    for t in engine.closed_trades:
        if t.holding_duration_sec >= 900.0:
            long_held_trades.append({
                "cycle": arch, "side": t.side, "dur": t.holding_duration_sec,
                "net": t.net_return_bps, "reason": t.exit_reason
            })

print(f"Total trades held >= 900s: {len(long_held_trades)}")
for t in long_held_trades:
    print(t)
