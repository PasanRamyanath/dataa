import os
import glob
import json
import collections
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

base_dir = "D:/PrivateProjects/BTC Futures/scratch"
archives = sorted([d for d in os.listdir(base_dir) if (d.startswith("archive_cycle") or d.startswith("archive_v3")) and os.path.isdir(os.path.join(base_dir, d))])

def test_exact_cycle48_suite():
    # Test the exact Cycle 48 engine across all 64 archives:
    # 1. Quad-Horizon Concordance: d_micro > 0, d_tact > 0, d_meso >= 0, d_macro > -0.05
    # 2. Conformal Certainty: conf_tact >= 0.42 and conf_meso >= 0.40 OR edge >= 10.0
    # 3. OFI Depth Guard: if depth available, bid_qty / ask_qty >= 1.02
    # 4. Stepped Ratchet Ladder (C48 + C43 + C60 TP)
    # 5. Fix 1: Taker SL (-35b or -45b)
    # 6. Fix 3: Runway Freeze (max_entry_bar = 270)
    
    maker_fee = 2.0
    taker_fee = 4.5
    
    sl_options = [35.0, 45.0, 9999.0]
    
    print("="*105)
    print("CYCLE 48 PEAK ENGINE SUITE ACROSS ALL 64 HISTORICAL CYCLES")
    print("="*105)
    print(f"{'SL Level':<20} | {'Trades':<6} | {'Win Rate':<8} | {'Net Realized PnL':<16} | {'Avg/Tr':<8} | {'100% WR Cycles':<14}")
    print("-" * 105)
    
    for sl in sl_options:
        total_trades = []
        active_cycles = 0
        wr100_cycles = 0
        profitable_cycles = 0
        reasons = collections.Counter()
        
        for arch in archives:
            ledgers = glob.glob(os.path.join(base_dir, arch, "*ledger*.jsonl"))
            if not ledgers: continue
            with open(ledgers[0], "r", encoding="utf-8") as f:
                bars = [json.loads(l) for l in f if l.strip().startswith("{")]
                
            rolling_15m = collections.deque(maxlen=90)
            recent_mids = collections.deque(maxlen=30)
            pos = None
            closed = []
            cum_pnl = 0.0
            last_exit_ts = 0; last_exit_px = 0.0; last_exit_was_win = True
            
            for idx, bar in enumerate(bars):
                mid = bar.get("bbo_mid", bar.get("mid", 0.0))
                bid = bar.get("bbo_bid", mid - 0.05)
                ask = bar.get("bbo_ask", mid + 0.05)
                bid_qty = bar.get("bbo_bid_qty", 10.0)
                ask_qty = bar.get("bbo_ask_qty", 10.0)
                ts_ms = bar.get("ts_ms", idx * 10000)
                rolling_15m.append(mid)
                recent_mids.append((ts_ms, mid))
                
                # Position management
                if pos is not None:
                    epx = pos["entry_px"]
                    gross = (bid - epx)/epx*10000.0 # Long-Only
                    pos["mfe"] = max(pos["mfe"], gross)
                    pos["mae"] = min(pos["mae"], gross)
                    mfe = pos["mfe"]; dur = (ts_ms - pos["ts"]) / 1000.0
                    
                    floor_gross = None
                    tier_name = None
                    if mfe >= 15.0: floor_gross = 12.50; tier_name = "TIER3_RUNNER"
                    elif mfe >= 10.0: floor_gross = 8.50; tier_name = "TIER2_RUNNER"
                    elif mfe >= 6.0: floor_gross = 6.00; tier_name = "TIER1_STEP"
                    elif mfe >= 4.5: floor_gross = 5.25; tier_name = "TIER0_5_C48"
                    elif mfe >= 3.8: floor_gross = 4.15; tier_name = "TIER0_SHIELD"
                    
                    # 1. Target TP (+35.0b)
                    if gross >= 35.0:
                        net = gross - 4.0
                        rec = {"cycle": arch, "net": net, "dur": dur, "mfe": mfe, "mae": pos["mae"], "reason": "TP"}
                        closed.append(rec); total_trades.append(rec); cum_pnl += net; reasons["TP"] += 1
                        last_exit_ts = ts_ms; last_exit_px = mid; last_exit_was_win = True; pos = None; continue
                    # 2. Ratchet Retrace Floor
                    elif floor_gross is not None and gross <= floor_gross:
                        net = floor_gross - 4.0
                        rec = {"cycle": arch, "net": net, "dur": dur, "mfe": mfe, "mae": pos["mae"], "reason": tier_name}
                        closed.append(rec); total_trades.append(rec); cum_pnl += net; reasons[tier_name] += 1
                        last_exit_ts = ts_ms; last_exit_px = mid; last_exit_was_win = True; pos = None; continue
                    # 3. Profitable Time Unwind (>= 900s & gross >= 4.0b)
                    elif dur >= 900.0 and gross >= 4.0:
                        net = gross - 4.0
                        rec = {"cycle": arch, "net": net, "dur": dur, "mfe": mfe, "mae": pos["mae"], "reason": "PROFIT_UNWIND"}
                        closed.append(rec); total_trades.append(rec); cum_pnl += net; reasons["PROFIT_UNWIND"] += 1
                        last_exit_ts = ts_ms; last_exit_px = mid; last_exit_was_win = True; pos = None; continue
                    # 4. Emergency Stop Loss
                    elif gross <= -sl:
                        actual_loss = min(-sl, gross)
                        net = actual_loss - (maker_fee + taker_fee)
                        rec = {"cycle": arch, "net": net, "dur": dur, "mfe": mfe, "mae": pos["mae"], "reason": "STOP_LOSS"}
                        closed.append(rec); total_trades.append(rec); cum_pnl += net; reasons["STOP_LOSS"] += 1
                        last_exit_ts = ts_ms; last_exit_px = mid; last_exit_was_win = False; pos = None; continue

                # Entries (IDLE state, max_entry_bar <= 270)
                if pos is None and idx >= 15 and idx <= 270:
                    tact = bar.get("tactical_probs", [0.33, 0.33, 0.33])
                    meso = bar.get("meso_probs", [0.33, 0.33, 0.33])
                    micro = bar.get("micro_probs", [0.33, 0.33, 0.33])
                    macro = bar.get("macro_probs", [0.33, 0.33, 0.33])
                    edge = bar.get("expected_net_edge_bps", bar.get("edge", 0.0))
                    
                    d_tact = tact[2] - tact[0]
                    d_micro = micro[2] - micro[0]
                    d_meso = meso[2] - meso[0]
                    d_macro = macro[2] - macro[0]
                    
                    # 1. Quad-Horizon Concordance
                    quad_long = (d_micro > 0.0) and (d_tact > 0.0) and (d_meso >= 0.0) and (d_macro > -0.05)
                    if not quad_long: continue
                    
                    # 2. Conformal Confidence Gate
                    conf_tact = max(tact); conf_meso = max(meso)
                    has_conviction = (conf_tact >= 0.42 and conf_meso >= 0.40) or (d_tact >= 0.12 and d_meso >= 0.03 and edge >= 10.0)
                    if not has_conviction: continue
                    
                    # 3. Tactical Confirmation
                    tact_confirmed = (d_tact >= 0.10 and d_micro > 0.010 and d_meso >= 0.02)
                    if not tact_confirmed and not (edge >= 5.0 and d_meso >= 0.02): continue
                    
                    # 4. OFI Depth Guard (only if live depth was recorded in ledger)
                    if "bbo_bid_qty" in bar and bar.get("bbo_ask_qty", 0) > 0:
                        imbalance = bar["bbo_bid_qty"] / bar["bbo_ask_qty"]
                        if imbalance < 1.02: continue
                        
                    # 5. Range Boundary Guard ($8.0 dollars from session high)
                    if len(rolling_15m) >= 18:
                        sess_hi = max(rolling_15m)
                        if (sess_hi - mid) < 8.0: continue
                        
                    # 6. Retrace / Cooldown barrier
                    if last_exit_ts > 0:
                        elapsed = (ts_ms - last_exit_ts) / 1000.0
                        if not last_exit_was_win and elapsed < 120.0: continue
                        if last_exit_was_win and elapsed < 60.0 and mid > (last_exit_px - 8.0): continue
                        
                    pos = {"entry_px": ask, "bar": idx, "ts": ts_ms, "mfe": 0.0, "mae": 0.0}

            # Session-end mark
            if pos is not None:
                epx = pos["entry_px"]
                gross = (bid - epx)/epx*10000.0
                net = gross - 4.0
                dur = (bars[-1].get("ts_ms", len(bars)*10000) - pos["ts"]) / 1000.0
                rec = {"cycle": arch, "net": net, "dur": dur, "mfe": pos["mfe"], "mae": pos["mae"], "reason": "SESSION_END_MARK"}
                closed.append(rec); total_trades.append(rec); cum_pnl += net; reasons["SESSION_END_MARK"] += 1

            if closed:
                active_cycles += 1
                w = sum(1 for c in closed if c["net"] > 0)
                if w == len(closed): wr100_cycles += 1
                if cum_pnl > 0: profitable_cycles += 1

        tot_w = sum(1 for t in total_trades if t["net"] > 0)
        tot_pnl = sum(t["net"] for t in total_trades)
        wr = (tot_w / len(total_trades) * 100.0) if total_trades else 0.0
        avg = (tot_pnl / len(total_trades)) if total_trades else 0.0
        label = f"Taker SL -{sl:.1f}b" if sl < 9999 else "Pure Mean Reversion"
        print(f"{label:<20} | {len(total_trades):<6} | {wr:>6.2f}% | {tot_pnl:>+12.2f} bps | {avg:>+6.2f}b | {wr100_cycles}/{active_cycles} ({wr100_cycles/max(1,active_cycles)*100:.1f}%)")
        print(f"  Reasons: {dict(reasons)}")
    print("="*105)

if __name__ == "__main__":
    test_exact_cycle48_suite()
