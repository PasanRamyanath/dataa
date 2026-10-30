import os
import glob
import json
import io
import sys
import types

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.append('feature engineering scripts')

from execution.champion_challenger_framework import Cycle66_QuadModeMultiTranchePeakEngine

base_dir = "D:/PrivateProjects/BTC Futures/scratch"
loss_cycles = ['archive_cycle31', 'archive_cycle38', 'archive_cycle39', 'archive_cycle55']

for arch in loss_cycles:
    ledgers = glob.glob(os.path.join(base_dir, arch, "*ledger*.jsonl"))
    if not ledgers:
        continue
    with open(ledgers[0], "r", encoding="utf-8") as f:
        bars = [json.loads(line) for line in f if line.strip().startswith("{")]

    engine = Cycle66_QuadModeMultiTranchePeakEngine(symbol="BTCUSDT")

    print(f"\n{'='*70}\nINSPECTING {arch} (Total Bars: {len(bars)})\n{'='*70}")

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

        prev_pos = engine.active_position
        prev_tranches = engine.pos_tranches

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

        if prev_pos is None and engine.active_position is not None:
            print(f"[BAR {idx:3d}] ENTRY: Side={engine.active_position.side} Price={engine.pos_orig_entry_price:.2f} Edge={edge:.2f} Dir={p_dir} Tact={tact} Meso={meso} Macro={macro}")
        
        if prev_pos is not None and engine.pos_tranches > prev_tranches:
            print(f"[BAR {idx:3d}] TRANCHE {engine.pos_tranches} FILLED: Blended Avg Price={engine.pos_avg_entry_price:.2f} Weight={engine.pos_weight:.1f}")

    if engine.active_position is not None:
        pos = engine.active_position
        last_mid = bars[-1].get("bbo_mid", bars[-1].get("mid", 0.0))
        sign = 1.0 if pos.side == "BUY" else -1.0
        u_bps = sign * ((last_mid - engine.pos_avg_entry_price) / engine.pos_avg_entry_price) * 1e4
        net_m2m = u_bps - 4.0
        print(f"[SESSION END] UNCLOSED POSITION: Side={pos.side} Tranches={engine.pos_tranches} AvgEP={engine.pos_avg_entry_price:.2f} LastMid={last_mid:.2f} MFE={engine.pos_mfe_bps:.2f} MAE={engine.pos_mae_bps:.2f} Gross={u_bps:.2f}b Net={net_m2m:.2f}b")

    for t in engine.closed_trades:
        print(f"[TRADE CLOSED] Side={t.side} Entry={t.entry_price:.2f} Exit={t.exit_price:.2f} Gross={t.gross_return_bps:.2f}b Net={t.net_return_bps:.2f}b Reason={t.exit_reason} Dur={t.holding_duration_sec:.0f}s")
