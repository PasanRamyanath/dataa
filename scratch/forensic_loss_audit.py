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
target_cycles = ['archive_cycle31', 'archive_cycle39', 'archive_cycle55']

for arch in target_cycles:
    ledgers = glob.glob(os.path.join(base_dir, arch, "*ledger*.jsonl"))
    if not ledgers:
        continue
    with open(ledgers[0], "r", encoding="utf-8") as f:
        bars = [json.loads(line) for line in f if line.strip().startswith("{")]

    engine = Cycle66_QuadModeMultiTranchePeakEngine(symbol="BTCUSDT")

    print("\n" + "=" * 90)
    print(f"DEEP FORENSIC AUDIT: {arch} (Total Bars in Ledger: {len(bars)})")
    print("=" * 90)

    entry_bar = -1
    exit_bar = -1
    loss_trade_info = None

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

        if prev_pos is None and engine.active_position is not None:
            entry_bar = idx
            print(f"[BAR {idx:3d} | ENTRY] Side={engine.active_position.side} Price={engine.pos_orig_entry_price:.2f} | Dir={p_dir} | Edge={edge:.2f} | Tact={tact} | Meso={meso} | Macro={macro}")

        if prev_pos is not None and engine.pos_tranches > prev_t:
            u_orig = (mid - engine.pos_orig_entry_price)/engine.pos_orig_entry_price * 1e4 * (1.0 if engine.active_position.side=="BUY" else -1.0)
            print(f"[BAR {idx:3d} | TRANCHE {engine.pos_tranches}] Mid={mid:.2f} | Excursion={u_orig:+.2f}b | New Blended EP={engine.pos_avg_entry_price:.2f} | Weight={engine.pos_weight:.1f}")

    if engine.active_position is not None:
        pos = engine.active_position
        last_mid = bars[-1].get("bbo_mid", bars[-1].get("mid", 0.0))
        sign = 1.0 if pos.side == "BUY" else -1.0
        orig_u = sign * ((last_mid - engine.pos_orig_entry_price) / engine.pos_orig_entry_price) * 1e4
        blend_u = sign * ((last_mid - engine.pos_avg_entry_price) / engine.pos_avg_entry_price) * 1e4
        net_m2m = blend_u - 4.0
        print(f"\n[SESSION END CUTOFF AT BAR {len(bars)-1}] Position Still Open!")
        print(f"  * Side: {pos.side}")
        print(f"  * Tranches Accumulated: {engine.pos_tranches} (Total Weight: {engine.pos_weight:.1f}x)")
        print(f"  * Original Entry Price: {engine.pos_orig_entry_price:.2f}")
        print(f"  * Blended Entry Price : {engine.pos_avg_entry_price:.2f}")
        print(f"  * Session Closing Price: {last_mid:.2f}")
        print(f"  * Peak Favorable Excursion (MFE): {engine.pos_mfe_bps:+.2f} bps")
        print(f"  * Max Adverse Excursion (MAE)   : {engine.pos_mae_bps:+.2f} bps")
        print(f"  * Excursion vs Original Entry  : {orig_u:+.2f} bps")
        print(f"  * Excursion vs Blended Entry   : {blend_u:+.2f} bps (GROSS PnL)")
        print(f"  * Honest Mark-to-Market Net PnL: {net_m2m:+.2f} bps (debiting 4.00 bps fee)")
        print(f"  * Bars Held: {len(bars) - 1 - entry_bar} bars ({(len(bars) - 1 - entry_bar)*10} seconds = {(len(bars) - 1 - entry_bar)*10/60:.1f} minutes)")

    for t in engine.closed_trades:
        print(f"[EARLIER CLOSED TRADE IN SESSION] Side={t.side} Entry={t.entry_price:.2f} Exit={t.exit_price:.2f} Gross={t.gross_return_bps:+.2f}b Net={t.net_return_bps:+.2f}b Reason={t.exit_reason}")
