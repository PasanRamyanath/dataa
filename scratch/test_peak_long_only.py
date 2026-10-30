import os
import glob
import json
import collections
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

base_dir = "D:/PrivateProjects/BTC Futures/scratch"
archives = sorted([d for d in os.listdir(base_dir) if (d.startswith("archive_cycle") or d.startswith("archive_v3")) and os.path.isdir(os.path.join(base_dir, d))])

def test_peak_long_only():
    maker_fee = 2.0
    taker_fee = 4.5
    
    # We test:
    # 1. Unchoked Ladder with Target TP (+35.0b)
    # 2. SL at -35b, -40b, -45b, and No SL
    # 3. Holding duration limit: 20m, 25m, 30m, Unlimited
    
    experiments = [
        {"name": "1. Peak Long-Only: SL -45b Taker, Max Entry B270, Hold 25m", "sl": 45.0, "max_dur": 1500.0, "max_bar": 270},
        {"name": "2. Peak Long-Only: SL -40b Taker, Max Entry B270, Hold 25m", "sl": 40.0, "max_dur": 1500.0, "max_bar": 270},
        {"name": "3. Peak Long-Only: SL -35b Taker, Max Entry B270, Hold 25m", "sl": 35.0, "max_dur": 1500.0, "max_bar": 270},
        {"name": "4. Peak Long-Only: SL -45b Taker, Max Entry B270, Hold 30m", "sl": 45.0, "max_dur": 1800.0, "max_bar": 270},
        {"name": "5. Peak Long-Only: SL -45b Taker, Max Entry B240, Hold 25m", "sl": 45.0, "max_dur": 1500.0, "max_bar": 240},
    ]
    
    ladder = [(35.0, 35.0, "TP"), (20.0, 16.00, "TIER4_EXP"), (14.0, 11.00, "TIER3_RUN"), (9.0, 7.00, "TIER2_MID"), (6.5, 5.25, "TIER1_C48"), (5.0, 4.15, "TIER0_BE")]
    
    print("="*105)
    print("PEAK LONG-ONLY ARCHITECTURE SWEEP ACROSS ALL 64 HISTORICAL CYCLES")
    print("="*105)
    print(f"{'Configuration':<52} | {'Trades':<6} | {'Win Rate':<8} | {'Net Realized PnL':<16} | {'Avg/Tr':<8} | {'100% WR Cycles':<14}")
    print("-" * 105)
    
    for exp in experiments:
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
            pos = None
            closed = []
            cum_pnl = 0.0
            last_exit_ts = 0; last_exit_px = 0.0
            
            for idx, bar in enumerate(bars):
                mid = bar.get("bbo_mid", bar.get("mid", 0.0))
                bid = bar.get("bbo_bid", mid - 0.05)
                ask = bar.get("bbo_ask", mid + 0.05)
                ts_ms = bar.get("ts_ms", idx * 10000)
                rolling_15m.append(mid)
                
                # Position management (LONG-ONLY)
                if pos is not None:
                    epx = pos["entry_px"]
                    gross = (bid - epx)/epx*10000.0
                    pos["mfe"] = max(pos["mfe"], gross)
                    pos["mae"] = min(pos["mae"], gross)
                    mfe = pos["mfe"]; dur = (ts_ms - pos["ts"]) / 1000.0
                    
                    # 1. Target TP
                    if gross >= 35.0:
                        net = gross - 4.0
                        rec = {"net": net, "dur": dur, "reason": "TP"}
                        closed.append(rec); total_trades.append(rec); cum_pnl += net; reasons["TP"] += 1
                        last_exit_ts = ts_ms; last_exit_px = mid; pos = None; continue
                        
                    # Find highest active floor
                    floor_gross = None
                    tier_name = None
                    for hurdle, floor, name in ladder:
                        if name == "TP": continue
                        if mfe >= hurdle:
                            floor_gross = floor
                            tier_name = name
                            break
                            
                    # 2. Ratchet Retrace
                    if floor_gross is not None and gross <= floor_gross:
                        net = floor_gross - 4.0
                        rec = {"net": net, "dur": dur, "reason": tier_name}
                        closed.append(rec); total_trades.append(rec); cum_pnl += net; reasons[tier_name] += 1
                        last_exit_ts = ts_ms; last_exit_px = mid; pos = None; continue
                        
                    # 3. Profitable Unwind (dur >= 900s & gross >= 4.0b)
                    elif dur >= 900.0 and gross >= 4.0:
                        net = gross - 4.0
                        rec = {"net": net, "dur": dur, "reason": "PROFIT_UNWIND"}
                        closed.append(rec); total_trades.append(rec); cum_pnl += net; reasons["PROFIT_UNWIND"] += 1
                        last_exit_ts = ts_ms; last_exit_px = mid; pos = None; continue
                        
                    # 4. Max Duration Limit
                    elif dur >= exp["max_dur"]:
                        net = gross - 4.0
                        rec = {"net": net, "dur": dur, "reason": "MAX_DUR_CUT"}
                        closed.append(rec); total_trades.append(rec); cum_pnl += net; reasons["MAX_DUR_CUT"] += 1
                        last_exit_ts = ts_ms; last_exit_px = mid; pos = None; continue
                        
                    # 5. Catastrophic Taker Breaker
                    elif gross <= -exp["sl"]:
                        actual_loss = min(-exp["sl"], gross)
                        net = actual_loss - (maker_fee + taker_fee)
                        rec = {"net": net, "dur": dur, "reason": "BREAKER"}
                        closed.append(rec); total_trades.append(rec); cum_pnl += net; reasons["BREAKER"] += 1
                        last_exit_ts = ts_ms; last_exit_px = mid; pos = None; continue

                # Entries (IDLE state, max_entry_bar <= exp['max_bar'])
                if pos is None and idx >= 15 and idx <= exp["max_bar"]:
                    p_dir = bar.get("primary_direction", bar.get("direction", 1))
                    edge = bar.get("expected_net_edge_bps", bar.get("edge", 0.0))
                    if p_dir != 2 or edge < 8.0: continue
                    
                    tact = bar.get("tactical_probs", [0.33, 0.33, 0.33])
                    meso = bar.get("meso_probs", [0.33, 0.33, 0.33])
                    micro = bar.get("micro_probs", [0.33, 0.33, 0.33])
                    macro = bar.get("macro_probs", [0.33, 0.33, 0.33])
                    
                    d_tact = tact[2] - tact[0]
                    d_meso = meso[2] - meso[0]
                    d_micro = micro[2] - micro[0]
                    d_macro = macro[2] - macro[0]
                    
                    # Quad-horizon agreement
                    if d_micro <= 0.0 or d_tact <= 0.05 or d_meso < 0.01 or d_macro < -0.05: continue
                    
                    # Quantile Range Guard (0.20 to 0.75)
                    if len(rolling_15m) >= 30:
                        hi = max(rolling_15m); lo = min(rolling_15m); rng = hi - lo
                        if rng >= 20.0:
                            pos_in_rng = (mid - lo) / rng
                            if pos_in_rng >= 0.75: continue
                            
                    # Retrace barrier
                    if last_exit_ts > 0 and (ts_ms - last_exit_ts) / 1000.0 < 120.0:
                        if mid > (last_exit_px - 8.0): continue
                        
                    pos = {"entry_px": ask, "bar": idx, "ts": ts_ms, "mfe": 0.0, "mae": 0.0}

            # Session-end mark
            if pos is not None:
                epx = pos["entry_px"]
                gross = (bid - epx)/epx*10000.0
                net = gross - 4.0
                dur = (bars[-1].get("ts_ms", len(bars)*10000) - pos["ts"]) / 1000.0
                rec = {"net": net, "dur": dur, "reason": "SESSION_END_MARK"}
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
        print(f"{exp['name']:<52} | {len(total_trades):<6} | {wr:>6.2f}% | {tot_pnl:>+12.2f} bps | {avg:>+6.2f}b | {wr100_cycles}/{active_cycles} ({wr100_cycles/max(1,active_cycles)*100:.1f}%)")
        print(f"  Reasons: {dict(reasons)}")
    print("="*105)

if __name__ == "__main__":
    test_peak_long_only()
