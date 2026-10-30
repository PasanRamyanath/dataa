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
archives = sorted([d for d in os.listdir(base_dir) if (d.startswith("archive_cycle") or d.startswith("archive_v3")) and os.path.isdir(os.path.join(base_dir, d))])

for min_scale_rem in [15, 25, 35, 45, 60]:
    all_closed_trades = []
    loss_trades = []

    for arch in archives:
        ledgers = glob.glob(os.path.join(base_dir, arch, "*ledger*.jsonl"))
        if not ledgers:
            continue
        with open(ledgers[0], "r", encoding="utf-8") as f:
            bars = [json.loads(line) for line in f if line.strip().startswith("{")]

        engine = Cycle66_QuadModeMultiTranchePeakEngine(symbol="BTCUSDT")

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

            # Custom evaluate tranche and ratchet exit with min_scale_rem
            # Micro tick
            now_ms = ts_ms * 1000 if ts_ms < 10**12 else ts_ms
            engine.ticks_seen = idx + 1
            engine.mid_history.append(mid)

            # Scale-in and exit logic with custom rem_bars
            if engine.active_position is not None:
                pos = engine.active_position
                sign = 1.0 if pos.side == "BUY" else -1.0
                rem_bars = max(0, len(bars) - idx)
                orig_u_bps = sign * ((mid - engine.pos_orig_entry_price) / engine.pos_orig_entry_price) * 1e4

                while engine.pos_tranches - 1 < len(engine.tranche_levels):
                    lvl_bps, add_w = engine.tranche_levels[engine.pos_tranches - 1]
                    if orig_u_bps <= lvl_bps and rem_bars >= min_scale_rem:
                        fill_px = bid if sign == 1.0 else ask
                        new_w = engine.pos_weight + add_w
                        engine.pos_avg_entry_price = (engine.pos_avg_entry_price * engine.pos_weight + fill_px * add_w) / new_w
                        pos.entry_price = engine.pos_avg_entry_price
                        pos.quantity = round(pos.quantity * (new_w / engine.pos_weight), 3)
                        engine.pos_weight = new_w
                        engine.pos_tranches += 1
                        engine.pos_mfe_bps = 0.0
                    else:
                        break

                u_bps = sign * ((mid - engine.pos_avg_entry_price) / engine.pos_avg_entry_price) * 1e4
                engine.pos_mfe_bps = max(engine.pos_mfe_bps, u_bps)
                engine.pos_mae_bps = min(engine.pos_mae_bps, u_bps)
                elapsed_s = (now_ms - pos.entry_ts_ms) / 1000.0

                active_floor, reason = (None, "")
                if engine.pos_tranches > 1:
                    if engine.pos_mfe_bps >= 12.00:
                        active_floor, reason = (9.50, "SCALE_R2_MAKER_TP")
                    elif engine.pos_mfe_bps >= 8.00:
                        active_floor, reason = (6.50, "SCALE_R1_MAKER_TP")
                    elif engine.pos_mfe_bps >= 5.50:
                        active_floor, reason = (4.60, "SCALE_BE_SHIELD")
                    elif engine.pos_mfe_bps >= 4.15 and elapsed_s >= 120.0:
                        active_floor, reason = (4.08, "SCALE_LATE_BE_SHIELD")
                else:
                    if engine.pos_mfe_bps >= 22.00:
                        active_floor, reason = (18.50, "TIER3_RUNNER_EXPANSION_TP")
                    elif engine.pos_mfe_bps >= 16.00:
                        active_floor, reason = (13.00, "TIER2_RUNNER_RATCHET_TP")
                    elif engine.pos_mfe_bps >= 11.00:
                        active_floor, reason = (8.50, "TIER1_STEPPED_RATCHET_TP")
                    elif engine.pos_mfe_bps >= 6.50:
                        active_floor, reason = (5.25, "TIER0_5_CYCLE48_CROWN_RATCHET_TP")
                    elif engine.pos_mfe_bps >= 4.15 and elapsed_s >= 180.0:
                        active_floor, reason = (4.08, "TIER0_LATE_BREAKEVEN_SHIELD")

                if u_bps >= 35.00:
                    engine._close_position(exit_price=engine.pos_avg_entry_price * (1.0 + sign * 0.0035), exit_ts_ms=now_ms, exit_fee_bps=2.0, reason="TARGET_MAKER_TP")
                elif active_floor is not None and u_bps <= active_floor:
                    engine._close_position(exit_price=engine.pos_avg_entry_price * (1.0 + sign * active_floor * 1e-4), exit_ts_ms=now_ms, exit_fee_bps=2.0, reason=reason)
                elif engine.pos_tranches > 1 and u_bps >= 6.50:
                    engine._close_position(exit_price=engine.pos_avg_entry_price * (1.0 + sign * u_bps * 1e-4), exit_ts_ms=now_ms, exit_fee_bps=2.0, reason="SCALE_TARGET_MAKER_TP")
                elif (elapsed_s >= 900.0 and u_bps >= 4.15) or (rem_bars <= 3 and u_bps >= 4.02):
                    engine._close_position(exit_price=engine.pos_avg_entry_price * (1.0 + sign * u_bps * 1e-4), exit_ts_ms=now_ms, exit_fee_bps=2.0, reason="PASSIVE_MAKER_TIME_UNWIND")
                elif orig_u_bps <= -45.00:
                    engine._close_position(exit_price=bid if sign == 1.0 else ask, exit_ts_ms=now_ms, exit_fee_bps=5.0, reason="HARD_STOP_LOSS_TAKER")

            # Gating
            if engine.state == "IDLE":
                if len(engine.mid_history) >= 11 and idx <= 270 and edge <= 48.0:
                    window = engine.mid_history[-91:]
                    sh = max(window)
                    sl = min(window)
                    rng = sh - sl
                    if rng >= 10.0:
                        rbps = (rng / mid) * 1e4
                        for sign in (1.0, -1.0):
                            if sign == -1.0 and rbps < 12.0:
                                continue
                            dt = sign * (tact[2] - tact[0])
                            dm = sign * (meso[2] - meso[0])
                            dmac = sign * (macro[2] - macro[0])
                            dmic = sign * (micro[2] - micro[0])
                            pct = (mid - sl) / rng if sign == 1.0 else (sh - mid) / rng

                            r60 = sign * ((mid - engine.mid_history[-7]) / engine.mid_history[-7]) * 1e4 if len(engine.mid_history) >= 7 else 0.0
                            r180 = sign * ((mid - engine.mid_history[-19]) / engine.mid_history[-19]) * 1e4 if len(engine.mid_history) >= 19 else 0.0
                            r300 = sign * ((mid - engine.mid_history[-31]) / engine.mid_history[-31]) * 1e4 if len(engine.mid_history) >= 31 else 0.0

                            p_ok = (p_dir == 2 and sign == 1.0) or (p_dir == 0 and sign == -1.0)

                            m1 = (pct <= 0.40 and rbps >= 8.0 and dm >= 0.08 and dmac >= 0.02 and dmic >= -0.02)
                            m2 = (p_ok and edge >= 2.0 and pct <= 0.74 and dt >= 0.05 and dm >= 0.02 and dmac >= -0.02 and dmic >= -0.02 and r60 <= 8.0 and r180 <= (99.0 if sign == 1.0 else 8.0) and r300 <= 12.0)
                            m3 = (pct <= 0.72 and rbps >= 6.0 and dt >= 0.10 and (dm >= 0.08 if not p_ok else dm >= 0.03) and dmac >= 0.00 and dmic >= 0.00 and r60 <= 5.5 and r180 <= 8.0 and r300 <= 8.0)
                            m4a = (sign == 1.0 and p_ok and idx <= 240 and pct <= 0.96 and rbps >= 14.0 and dt >= 0.16 and dm >= 0.195 and dmac >= 0.05 and dmic >= 0.00 and r60 <= 5.5 and r300 <= 12.0)
                            m4b = (sign == 1.0 and p_ok and idx <= 240 and pct <= 0.96 and rbps >= 7.5 and dt >= 0.16 and dm >= 0.08 and dmac >= 0.30 and dmic >= 0.04 and r60 <= 5.5 and r300 <= 5.0)
                            m4 = m4a or m4b

                            if m1 or m2 or m3 or m4:
                                from execution.hierarchical_sniper_engine import ActiveSwingPosition
                                ep = bid if sign == 1.0 else ask
                                side = "BUY" if sign == 1.0 else "SELL"
                                engine.active_position = ActiveSwingPosition(
                                    symbol="BTCUSDT", side=side, entry_price=ep, quantity=0.01, entry_ts_ms=now_ms,
                                    target_holding_seconds=900, take_profit_bps=35.0, stop_loss_bps=45.0,
                                    tp_price=ep * (1.0 + sign * 0.0035), sl_price=ep * (1.0 - sign * 0.0045), maker_entry_fee_bps=2.0
                                )
                                engine.pos_orig_entry_price = ep
                                engine.pos_avg_entry_price = ep
                                engine.pos_weight = 1.0
                                engine.pos_tranches = 1
                                engine.pos_mfe_bps = 0.0
                                engine.pos_mae_bps = 0.0
                                engine.state = "FILLED_SWING_HOLDING"
                                break

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
    print(f"min_scale_rem={min_scale_rem:2d} -> Total={n:3d} | Wins={w:3d} | Losses={l:2d} | WR={wr:.2f}% | Net={tot_pnl:+.2f}b | Losses: {[t['cycle'] + '_' + t['side'] + '(' + str(round(t['net'], 1)) + 'b)' for t in loss_trades]}")
