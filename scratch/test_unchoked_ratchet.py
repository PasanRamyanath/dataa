import os
import glob
import json
import collections
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

base_dir = "D:/PrivateProjects/BTC Futures/scratch"
archives = sorted([d for d in os.listdir(base_dir) if (d.startswith("archive_cycle") or d.startswith("archive_v3")) and os.path.isdir(os.path.join(base_dir, d))])

def test_unchoked_ratchets():
    # Test calibrated unchoked ratchet ladders:
    # Baseline: choked ratchet where floor was higher than hurdle (choked on bar 1)
    # Calibrated 1: Proper retrace buffers (hurdle > floor)
    # Calibrated 2: Runner-biased (let runners expand to +15b, +25b, +35b)
    
    maker_fee = 2.0
    taker_fee = 4.5
    
    ratchet_configs = [
        {
            "name": "1. Old Choked Ladder (Hurdle <= Floor)",
            "ladder": [(35.0, 35.0, "TP"), (15.0, 12.50, "TIER3"), (10.0, 8.50, "TIER2"), (6.0, 6.25, "TIER1"), (4.5, 5.25, "TIER0_5"), (3.8, 4.15, "TIER0")]
        },
        {
            "name": "2. Unchoked Ladder (Hurdle > Floor, 0.85b Retrace Buffer)",
            # (hurdle, floor, name)
            "ladder": [(35.0, 35.0, "TP"), (20.0, 16.00, "TIER4_EXP"), (14.0, 11.00, "TIER3_RUN"), (9.0, 7.00, "TIER2_MID"), (6.5, 5.25, "TIER1_C48"), (5.0, 4.15, "TIER0_BE")]
        },
        {
            "name": "3. Dynamic Alpha Ladder (Allowing Runners to Target +35b)",
            "ladder": [(35.0, 35.0, "TP"), (22.0, 18.00, "TIER4_EXP"), (15.0, 12.00, "TIER3_RUN"), (10.0, 7.50, "TIER2_MID"), (7.0, 5.50, "TIER1_C48"), (5.2, 4.20, "TIER0_BE")]
        },
    ]
    
    print("="*105)
    print("UNCHOKED RATCHET LADDER EXPERIMENT (ALLOWING WINNERS TO EXPAND)")
    print("="*105)
    print(f"{'Ratchet Configuration':<48} | {'Trades':<6} | {'Win Rate':<8} | {'Net Realized PnL':<16} | {'Avg/Tr':<8} | {'100% WR Cycles':<14}")
    print("-" * 105)
    
    for cfg in ratchet_configs:
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
            last_exit_ts = 0; last_exit_side = ""; last_exit_px = 0.0
            
            for idx, bar in enumerate(bars):
                mid = bar.get("bbo_mid", bar.get("mid", 0.0))
                bid = bar.get("bbo_bid", mid - 0.05)
                ask = bar.get("bbo_ask", mid + 0.05)
                ts_ms = bar.get("ts_ms", idx * 10000)
                rolling_15m.append(mid)
                
                # Position management
                if pos is not None:
                    side = pos["side"]; epx = pos["entry_px"]
                    gross = (bid - epx)/epx*10000.0 if side == "BUY" else (epx - ask)/epx*10000.0
                    pos["mfe"] = max(pos["mfe"], gross)
                    pos["mae"] = min(pos["mae"], gross)
                    mfe = pos["mfe"]; dur = (ts_ms - pos["ts"]) / 1000.0
                    
                    # Full Target TP
                    if gross >= 35.0:
                        net = gross - 4.0
                        rec = {"net": net, "dur": dur, "reason": "TP"}
                        closed.append(rec); total_trades.append(rec); cum_pnl += net; reasons["TP"] += 1
                        last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue
                        
                    # Find highest active floor
                    floor_gross = None
                    tier_name = None
                    for hurdle, floor, name in cfg["ladder"]:
                        if name == "TP": continue
                        if mfe >= hurdle:
                            floor_gross = floor
                            tier_name = name
                            break
                            
                    # Retrace exit
                    if floor_gross is not None and gross <= floor_gross:
                        net = floor_gross - 4.0
                        rec = {"net": net, "dur": dur, "reason": tier_name}
                        closed.append(rec); total_trades.append(rec); cum_pnl += net; reasons[tier_name] += 1
                        last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue
                        
                    # Profitable Time Unwind (>= 900s & gross >= 4.0b)
                    elif dur >= 900.0 and gross >= 4.0:
                        net = gross - 4.0
                        rec = {"net": net, "dur": dur, "reason": "PROFIT_UNWIND"}
                        closed.append(rec); total_trades.append(rec); cum_pnl += net; reasons["PROFIT_UNWIND"] += 1
                        last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue
                        
                    # Hard duration cut at 25m (1500s)
                    elif dur >= 1500.0:
                        net = gross - 4.0
                        rec = {"net": net, "dur": dur, "reason": "HARD_DURATION_CUT"}
                        closed.append(rec); total_trades.append(rec); cum_pnl += net; reasons["HARD_DURATION_CUT"] += 1
                        last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue
                        
                    # Catastrophic Taker Breaker
                    elif gross <= -45.0:
                        actual_loss = min(-45.0, gross)
                        net = actual_loss - (maker_fee + taker_fee)
                        rec = {"net": net, "dur": dur, "reason": "BREAKER"}
                        closed.append(rec); total_trades.append(rec); cum_pnl += net; reasons["BREAKER"] += 1
                        last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue

                # Entries (max_entry_bar = 270)
                if pos is None and idx >= 15 and idx <= 270:
                    p_dir = bar.get("primary_direction", bar.get("direction", 1))
                    edge = bar.get("expected_net_edge_bps", bar.get("edge", 0.0))
                    if p_dir == 1 or edge < 8.0: continue
                    
                    tact = bar.get("tactical_probs", [0.33, 0.33, 0.33])
                    meso = bar.get("meso_probs", [0.33, 0.33, 0.33])
                    d_tact = tact[2] - tact[0]
                    d_meso = meso[2] - meso[0]
                    
                    cand_side = None
                    if p_dir == 2 and d_tact >= 0.05 and d_meso >= 0.01: cand_side = "BUY"
                    elif p_dir == 0 and d_tact <= -0.05 and d_meso <= -0.01: cand_side = "SELL"
                    
                    if cand_side is not None:
                        if len(rolling_15m) >= 30:
                            hi = max(rolling_15m); lo = min(rolling_15m); rng = hi - lo
                            if rng >= 20.0:
                                pos_in_rng = (mid - lo) / rng
                                if cand_side == "BUY" and pos_in_rng >= 0.75: continue
                                if cand_side == "SELL" and pos_in_rng <= 0.25: continue
                        if last_exit_ts > 0 and (ts_ms - last_exit_ts) / 1000.0 < 200.0:
                            if cand_side == "BUY" and last_exit_side == "BUY" and mid > (last_exit_px - 10.0): continue
                            if cand_side == "SELL" and last_exit_side == "SELL" and mid < (last_exit_px + 10.0): continue
                        pos = {"side": cand_side, "entry_px": ask if cand_side == "BUY" else bid, "bar": idx, "ts": ts_ms, "mfe": 0.0, "mae": 0.0}

            if pos is not None:
                side = pos["side"]; epx = pos["entry_px"]
                gross = (bid - epx)/epx*10000.0 if side == "BUY" else (epx - ask)/epx*10000.0
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
        print(f"{cfg['name']:<48} | {len(total_trades):<6} | {wr:>6.2f}% | {tot_pnl:>+12.2f} bps | {avg:>+6.2f}b | {wr100_cycles}/{active_cycles} ({wr100_cycles/max(1,active_cycles)*100:.1f}%)")
        print(f"  Reasons: {dict(reasons)}")
    print("="*105)

if __name__ == "__main__":
    test_unchoked_ratchets()
