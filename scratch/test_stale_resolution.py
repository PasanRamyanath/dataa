import os
import glob
import json
import collections
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

base_dir = "D:/PrivateProjects/BTC Futures/scratch"
archives = sorted([d for d in os.listdir(base_dir) if (d.startswith("archive_cycle") or d.startswith("archive_v3")) and os.path.isdir(os.path.join(base_dir, d))])

def test_stale_trade_resolution():
    # Let's test what happens if we add an Invalidation / Stale Trade Unwind:
    # Rule: If holding duration >= 900s (15 min) and peak MFE never reached +3.80 bps (dead trade),
    # what if we close it? Or what if macro model reverses against the position?
    
    experiments = [
        {"name": "1. Baseline (Current -45b SL, Unwind only if gross >= 4.0b)", "dead_dur": 99999, "sl_bps": 45.0},
        {"name": "2. Stale Cut at 15m (dur >= 900s if MFE < 3.8b)", "dead_dur": 900.0, "sl_bps": 45.0},
        {"name": "3. Stale Cut at 20m (dur >= 1200s if MFE < 3.8b)", "dead_dur": 1200.0, "sl_bps": 45.0},
        {"name": "4. Model Invalidation (if macro/meso strongly opposes pos)", "macro_inval": True, "sl_bps": 45.0},
        {"name": "5. Macro Inval + Stale Cut 15m + -45b SL", "macro_inval": True, "dead_dur": 900.0, "sl_bps": 45.0},
    ]
    
    for exp in experiments:
        maker_fee = 2.0
        taker_fee = 4.5
        total_trades = 0
        total_wins = 0
        total_net = 0.0
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
                
                tact = bar.get("tactical_probs", [0.33, 0.33, 0.33])
                meso = bar.get("meso_probs", [0.33, 0.33, 0.33])
                macro = bar.get("macro_probs", [0.33, 0.33, 0.33])
                d_macro = macro[2] - macro[0]
                d_meso = meso[2] - meso[0]
                
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
                    
                    # 1. TP
                    if gross >= 35.0:
                        net = gross - 4.0; closed.append(net); cum_pnl += net; reasons["TP"] += 1
                        last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue
                    # 2. Ratchet
                    elif floor_gross is not None and gross <= floor_gross:
                        net = floor_gross - 4.0; closed.append(net); cum_pnl += net; reasons[tier_name] += 1
                        last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue
                    # 3. Profitable Time Unwind
                    elif dur >= 900.0 and gross >= 4.0:
                        net = gross - 4.0; closed.append(net); cum_pnl += net; reasons["PROFIT_UNWIND"] += 1
                        last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue
                    # 4. Macro Model Invalidation Cut (if models turn strongly opposing)
                    elif exp.get("macro_inval") and mfe < 3.8:
                        is_opposed = (side == "BUY" and d_macro < -0.10 and d_meso < -0.05) or (side == "SELL" and d_macro > 0.10 and d_meso > 0.05)
                        if is_opposed and dur >= 180.0: # after at least 3 minutes
                            net = gross - (maker_fee + taker_fee)
                            closed.append(net); cum_pnl += net; reasons["MACRO_INVAL"] += 1
                            last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue
                    # 5. Stale Trade Dead Unwind (if held long without ever reaching +3.8b MFE)
                    elif dur >= exp.get("dead_dur", 99999) and mfe < 3.8:
                        net = gross - 4.0 # Maker exit
                        closed.append(net); cum_pnl += net; reasons["DEAD_UNWIND"] += 1
                        last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue
                    # 6. Catastrophic Breaker
                    elif gross <= -exp["sl_bps"]:
                        net = -exp["sl_bps"] - (maker_fee + taker_fee)
                        closed.append(net); cum_pnl += net; reasons["BREAKER"] += 1
                        last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue

                # Entries
                if pos is None and idx >= 15:
                    d_tact = tact[2] - tact[0]
                    micro = bar.get("micro_probs", [0.33, 0.33, 0.33])
                    d_micro = micro[2] - micro[0]
                    
                    is_long = (d_tact >= 0.08 and d_meso >= 0.015 and d_micro >= 0.01 and d_macro >= -0.02)
                    is_short = (d_tact <= -0.08 and d_meso <= -0.015 and d_micro <= -0.01 and d_macro <= 0.02)
                    cand_side = "BUY" if is_long else ("SELL" if is_short else None)
                    
                    if cand_side is not None:
                        if len(rolling_15m) >= 30:
                            hi = max(rolling_15m); lo = min(rolling_15m); rng = hi - lo
                            if rng >= 20.0:
                                pos_in_rng = (mid - lo) / rng
                                if cand_side == "BUY" and pos_in_rng >= 0.80: continue
                                if cand_side == "SELL" and pos_in_rng <= 0.20: continue
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
        print(f"\n{exp['name']}")
        print(f"  Trades: {total_trades} | Win Rate: {wr:.1f}% | Net PnL: {total_net:+.2f} bps | Avg: {total_net/total_trades:+.2f}b")
        print(f"  100% WR Cycles: {wr100_cycles}/{active_cycles} | Profitable Cycles: {profitable_cycles}/{active_cycles}")
        print(f"  Reasons: {dict(reasons)}")

if __name__ == "__main__":
    test_stale_trade_resolution()
