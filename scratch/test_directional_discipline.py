import os
import glob
import json
import collections
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

base_dir = "D:/PrivateProjects/BTC Futures/scratch"
archives = sorted([d for d in os.listdir(base_dir) if (d.startswith("archive_cycle") or d.startswith("archive_v3")) and os.path.isdir(os.path.join(base_dir, d))])

def test_peak_with_directional_discipline():
    # Test strict entry filters to prevent the 7 drop trades:
    # 1. Primary Direction must NOT be 1 (No trading when primary_direction == FLAT)
    # 2. Expected Net Edge >= 5.0 bps
    # 3. Quantile Range Guard: 0.25 <= pos_in_rng <= 0.75
    # 4. Multi-tier ratchets (C48 + C43 + C60)
    # 5. Stop Loss at -45.0 bps (Taker Emergency Breaker)
    
    maker_fee = 2.0
    taker_fee = 4.5
    
    experiments = [
        {"name": "A. Strict Directional + Edge>=5 + Quantile 0.25-0.75 + -45b Taker SL", "min_edge": 5.0, "sl_bps": 45.0, "q_lo": 0.25, "q_hi": 0.75, "no_flat": True},
        {"name": "B. Strict Directional + Edge>=8 + Quantile 0.20-0.80 + -45b Taker SL", "min_edge": 8.0, "sl_bps": 45.0, "q_lo": 0.20, "q_hi": 0.80, "no_flat": True},
        {"name": "C. Strict Directional + Edge>=5 + Quantile 0.25-0.75 + Pure Mean Reversion (No SL)", "min_edge": 5.0, "sl_bps": 9999.0, "q_lo": 0.25, "q_hi": 0.75, "no_flat": True},
    ]
    
    for exp in experiments:
        total_trades = 0
        total_wins = 0
        total_net = 0.0
        active_cycles = 0
        wr100_cycles = 0
        profitable_cycles = 0
        reasons = collections.Counter()
        losses = []
        
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
                    
                    floor_gross = None
                    tier_name = None
                    if mfe >= 15.0: floor_gross = 12.50; tier_name = "TIER3"
                    elif mfe >= 10.0: floor_gross = 8.50; tier_name = "TIER2"
                    elif mfe >= 6.0: floor_gross = 6.25; tier_name = "TIER1"
                    elif mfe >= 4.5: floor_gross = 5.25; tier_name = "TIER0_5"
                    elif mfe >= 3.8: floor_gross = 4.15; tier_name = "TIER0"
                    
                    if gross >= 35.0:
                        net = gross - 4.0; closed.append(net); cum_pnl += net; reasons["TP"] += 1
                        last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue
                    elif floor_gross is not None and gross <= floor_gross:
                        net = floor_gross - 4.0; closed.append(net); cum_pnl += net; reasons[tier_name] += 1
                        last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue
                    elif dur >= 900.0 and gross >= 4.0:
                        net = gross - 4.0; closed.append(net); cum_pnl += net; reasons["PROFIT_UNWIND"] += 1
                        last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue
                    elif gross <= -exp["sl_bps"]:
                        actual_loss = min(-exp["sl_bps"], gross)
                        net = actual_loss - (maker_fee + taker_fee)
                        closed.append(net); cum_pnl += net; reasons["BREAKER"] += 1
                        losses.append({"cycle": arch, "side": side, "net": net, "dur": dur, "mae": pos["mae"]})
                        last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue

                # Entries
                if pos is None and idx >= 15:
                    p_dir = bar.get("primary_direction", bar.get("direction", 1))
                    edge = bar.get("expected_net_edge_bps", bar.get("edge", 0.0))
                    
                    if exp.get("no_flat") and p_dir == 1:
                        continue
                    if edge < exp["min_edge"]:
                        continue
                        
                    tact = bar.get("tactical_probs", [0.33, 0.33, 0.33])
                    meso = bar.get("meso_probs", [0.33, 0.33, 0.33])
                    micro = bar.get("micro_probs", [0.33, 0.33, 0.33])
                    macro = bar.get("macro_probs", [0.33, 0.33, 0.33])
                    
                    d_tact = tact[2] - tact[0]
                    d_micro = micro[2] - micro[0]
                    d_meso = meso[2] - meso[0]
                    d_macro = macro[2] - macro[0]
                    
                    cand_side = None
                    if p_dir == 2 and d_tact >= 0.05 and d_meso >= 0.01:
                        cand_side = "BUY"
                    elif p_dir == 0 and d_tact <= -0.05 and d_meso <= -0.01:
                        cand_side = "SELL"
                        
                    if cand_side is not None:
                        if len(rolling_15m) >= 30:
                            hi = max(rolling_15m); lo = min(rolling_15m); rng = hi - lo
                            if rng >= 20.0:
                                pos_in_rng = (mid - lo) / rng
                                if cand_side == "BUY" and pos_in_rng >= exp["q_hi"]: continue
                                if cand_side == "SELL" and pos_in_rng <= exp["q_lo"]: continue
                        if last_exit_ts > 0 and (ts_ms - last_exit_ts) / 1000.0 < 200.0:
                            if cand_side == "BUY" and last_exit_side == "BUY" and mid > (last_exit_px - 10.0): continue
                            if cand_side == "SELL" and last_exit_side == "SELL" and mid < (last_exit_px + 10.0): continue
                        pos = {"side": cand_side, "entry_px": ask if cand_side == "BUY" else bid, "bar": idx, "ts": ts_ms, "mfe": 0.0, "mae": 0.0}

            if closed:
                active_cycles += 1
                w = len([c for c in closed if c > 0])
                total_trades += len(closed)
                total_wins += w
                total_net += cum_pnl
                if cum_pnl > 0: profitable_cycles += 1
                if w == len(closed): wr100_cycles += 1

        wr = (total_wins / total_trades * 100.0) if total_trades > 0 else 0.0
        avg = (total_net / total_trades) if total_trades > 0 else 0.0
        print(f"\nConfiguration: {exp['name']}")
        print(f"  Total Trades: {total_trades} across {active_cycles} active sessions")
        print(f"  Win Rate: {wr:.1f}% ({total_wins} Wins / {total_trades - total_wins} Losses)")
        print(f"  Net Realized PnL: {total_net:+.2f} bps | Avg Net/Trade: {avg:+.2f} bps")
        print(f"  100% WR Cycles: {wr100_cycles}/{active_cycles} ({wr100_cycles/max(1,active_cycles)*100:.1f}%)")
        print(f"  Profitable Cycles: {profitable_cycles}/{active_cycles} ({profitable_cycles/max(1,active_cycles)*100:.1f}%)")
        print(f"  Exit Reasons: {dict(reasons)}")
        if losses:
            print(f"  Losses breakdown ({len(losses)}):")
            for l in losses:
                print(f"    Cycle: {l['cycle']:<18} Side: {l['side']} Net: {l['net']:.2f}b Dur: {l['dur']:.1f}s MAE: {l['mae']:.2f}b")

if __name__ == "__main__":
    test_peak_with_directional_discipline()
