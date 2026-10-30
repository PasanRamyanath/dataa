import os
import glob
import json
import collections
import io
import sys

# Ensure UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

base_dir = "D:/PrivateProjects/BTC Futures/scratch"
archives = sorted([d for d in os.listdir(base_dir) if (d.startswith("archive_cycle") or d.startswith("archive_v3")) and os.path.isdir(os.path.join(base_dir, d))])

def test_smart_invalidation_matrix():
    # Test combinations of:
    # 1. Adverse + Dead MFE condition: gross <= -X and mfe < Y
    # 2. Macro flip condition: macro/meso flip direction while gross < 0
    # 3. Model reversal after 180s
    
    experiments = [
        {"name": "A. No Inval (Baseline with Fix 1 + Fix 3)", "type": "none"},
        {"name": "B. Adverse -15b with MFE < 2.0b", "type": "adverse_dead", "adv": 15.0, "mfe_cap": 2.0},
        {"name": "C. Adverse -20b with MFE < 2.0b", "type": "adverse_dead", "adv": 20.0, "mfe_cap": 2.0},
        {"name": "D. Adverse -25b with MFE < 2.5b", "type": "adverse_dead", "adv": 25.0, "mfe_cap": 2.5},
        {"name": "E. Adverse -30b with MFE < 3.0b", "type": "adverse_dead", "adv": 30.0, "mfe_cap": 3.0},
        {"name": "F. Macro Model Reversal (d_macro opposes pos by >=0.10 & dur>=180s)", "type": "macro_flip"},
        {"name": "G. Macro Reversal + Adverse -20b with MFE < 2.0b", "type": "combo", "adv": 20.0, "mfe_cap": 2.0},
    ]
    
    maker_fee = 2.0
    taker_fee = 4.5
    
    print("="*105)
    print("SMART ADVERSE & MODEL-BASED INVALIDATION SWEEP (FIX 1 & FIX 3 ACTIVE)")
    print("="*105)
    print(f"{'Experiment Name':<42} | {'Trades':<6} | {'Win Rate':<8} | {'Net Realized PnL':<16} | {'Avg/Tr':<8} | {'100% WR Cycles':<14}")
    print("-" * 105)
    
    for exp in experiments:
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
                
                macro = bar.get("macro_probs", [0.33, 0.33, 0.33])
                meso = bar.get("meso_probs", [0.33, 0.33, 0.33])
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
                    # 3. Profitable Unwind
                    elif dur >= 900.0 and gross >= 4.0:
                        net = gross - 4.0; closed.append(net); cum_pnl += net; reasons["PROFIT_UNWIND"] += 1
                        last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue
                        
                    # 4. Smart Invalidation
                    elif exp["type"] == "adverse_dead" and gross <= -exp["adv"] and mfe <= exp["mfe_cap"]:
                        actual_loss = min(-exp["adv"], gross)
                        net = actual_loss - (maker_fee + taker_fee)
                        closed.append(net); cum_pnl += net; reasons["DEAD_ADVERSE_INVAL"] += 1
                        last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue
                        
                    elif exp["type"] == "macro_flip" and dur >= 180.0 and gross < 0.0:
                        opposed = (side == "BUY" and d_macro < -0.10 and d_meso < -0.05) or (side == "SELL" and d_macro > 0.10 and d_meso > 0.05)
                        if opposed:
                            net = gross - (maker_fee + taker_fee)
                            closed.append(net); cum_pnl += net; reasons["MACRO_FLIP_INVAL"] += 1
                            last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue
                            
                    elif exp["type"] == "combo":
                        opposed = (side == "BUY" and d_macro < -0.10 and d_meso < -0.05) or (side == "SELL" and d_macro > 0.10 and d_meso > 0.05)
                        if opposed and dur >= 180.0 and gross < 0.0:
                            net = gross - (maker_fee + taker_fee)
                            closed.append(net); cum_pnl += net; reasons["MACRO_FLIP_INVAL"] += 1
                            last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue
                        elif gross <= -exp["adv"] and mfe <= exp["mfe_cap"]:
                            actual_loss = min(-exp["adv"], gross)
                            net = actual_loss - (maker_fee + taker_fee)
                            closed.append(net); cum_pnl += net; reasons["DEAD_ADVERSE_INVAL"] += 1
                            last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue

                    # 5. Catastrophic Breaker
                    elif gross <= -45.0:
                        actual_loss = min(-45.0, gross)
                        net = actual_loss - (maker_fee + taker_fee)
                        closed.append(net); cum_pnl += net; reasons["BREAKER"] += 1
                        last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue

                # Entries (IDLE state, max_bar=270)
                if pos is None and idx >= 15 and idx <= 270:
                    tact = bar.get("tactical_probs", [0.33, 0.33, 0.33])
                    micro = bar.get("micro_probs", [0.33, 0.33, 0.33])
                    d_tact = tact[2] - tact[0]
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

            if pos is not None:
                side = pos["side"]; epx = pos["entry_px"]
                gross = (bid - epx)/epx*10000.0 if side == "BUY" else (epx - ask)/epx*10000.0
                net = gross - (2 * maker_fee)
                closed.append(net); cum_pnl += net; reasons["SESSION_END_MARK"] += 1

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
        print(f"{exp['name']:<42} | {total_trades:<6} | {wr:>6.2f}% | {total_net:>+12.2f} bps | {avg:>+6.2f}b | {wr100_cycles}/{active_cycles} ({wr100_cycles/max(1,active_cycles)*100:.1f}%)")
        print(f"  Reasons: {dict(reasons)}")
    print("="*105)

if __name__ == "__main__":
    test_smart_invalidation_matrix()
