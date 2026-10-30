import os
import glob
import json
import collections
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

base_dir = "D:/PrivateProjects/BTC Futures/scratch"
archives = sorted([d for d in os.listdir(base_dir) if (d.startswith("archive_cycle") or d.startswith("archive_v3")) and os.path.isdir(os.path.join(base_dir, d))])

def analyze_drops_and_stop_loss():
    # We will test multiple stop loss configurations:
    # 1. Maker -45.0 bps
    # 2. Taker -45.0 bps (realistic emergency market stop with 4.5 bps taker fee)
    # 3. Taker -35.0 bps
    # 4. Taker -25.0 bps
    # 5. Taker -15.0 bps
    # 6. Pure Mean Reversion (No SL)
    
    sl_configs = [
        {"name": "No SL (Pure Mean Reversion)", "sl_bps": 9999.0, "is_taker": False},
        {"name": "Maker -45.0 bps (Current Code)", "sl_bps": 45.0, "is_taker": False},
        {"name": "Taker -45.0 bps (Aggressive Emergency Stop)", "sl_bps": 45.0, "is_taker": True},
        {"name": "Taker -35.0 bps", "sl_bps": 35.0, "is_taker": True},
        {"name": "Taker -25.0 bps", "sl_bps": 25.0, "is_taker": True},
        {"name": "Taker -15.0 bps (Tight Cut)", "sl_bps": 15.0, "is_taker": True},
    ]
    
    results = []
    
    for cfg in sl_configs:
        maker_fee = 2.0
        taker_fee = 4.5
        total_trades = 0
        total_wins = 0
        total_net = 0.0
        active_cycles = 0
        wr100_cycles = 0
        profitable_cycles = 0
        tier_counts = collections.Counter()
        all_trade_records = []
        
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
                    
                    # Track sudden drop velocity (max drop from peak MFE)
                    drop_from_peak = mfe - gross
                    pos["max_drop_from_peak"] = max(pos.get("max_drop_from_peak", 0.0), drop_from_peak)
                    
                    # Full Unified Ratchet Ladder
                    floor_gross = None
                    tier_name = None
                    if mfe >= 15.0:
                        floor_gross = 12.50
                        tier_name = "TIER3_RUNNER_EXPANSION"
                    elif mfe >= 10.0:
                        floor_gross = 8.50
                        tier_name = "TIER2_STEPPED_RATCHET"
                    elif mfe >= 6.0:
                        floor_gross = 6.25
                        tier_name = "TIER1_STEPPED_RATCHET"
                    elif mfe >= 4.5:
                        floor_gross = 5.25
                        tier_name = "TIER0_5_C48_RATCHET"
                    elif mfe >= 3.8:
                        floor_gross = 4.15
                        tier_name = "TIER0_SHIELD"
                        
                    # 1. Alpha TP (+35 bps)
                    if gross >= 35.0:
                        net = gross - (2 * maker_fee)
                        rec = {"cycle": arch, "side": side, "net": net, "dur": dur, "mfe": mfe, "mae": pos["mae"], "reason": "TARGET_TP", "max_drop": pos["max_drop_from_peak"]}
                        closed.append(net); cum_pnl += net; tier_counts["TARGET_TP"] += 1; all_trade_records.append(rec)
                        last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue
                    # 2. Ratchet Retrace Floor
                    elif floor_gross is not None and gross <= floor_gross:
                        net = floor_gross - (2 * maker_fee)
                        rec = {"cycle": arch, "side": side, "net": net, "dur": dur, "mfe": mfe, "mae": pos["mae"], "reason": tier_name, "max_drop": pos["max_drop_from_peak"]}
                        closed.append(net); cum_pnl += net; tier_counts[tier_name] += 1; all_trade_records.append(rec)
                        last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue
                    # 3. Profitable Time Unwind
                    elif dur >= 900.0 and gross >= 4.0:
                        net = gross - (2 * maker_fee)
                        rec = {"cycle": arch, "side": side, "net": net, "dur": dur, "mfe": mfe, "mae": pos["mae"], "reason": "PROFIT_UNWIND", "max_drop": pos["max_drop_from_peak"]}
                        closed.append(net); cum_pnl += net; tier_counts["PROFIT_UNWIND"] += 1; all_trade_records.append(rec)
                        last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue
                    # 4. Stop Loss / Catastrophic Breaker
                    elif gross <= -cfg["sl_bps"]:
                        exit_fee = taker_fee if cfg["is_taker"] else maker_fee
                        # In sudden drops, price can gap past the stop level in a bar
                        actual_loss_gross = min(-cfg["sl_bps"], gross)
                        net = actual_loss_gross - (maker_fee + exit_fee)
                        rec = {"cycle": arch, "side": side, "net": net, "dur": dur, "mfe": mfe, "mae": pos["mae"], "reason": "STOP_LOSS", "max_drop": pos["max_drop_from_peak"], "actual_gross": actual_loss_gross}
                        closed.append(net); cum_pnl += net; tier_counts["STOP_LOSS"] += 1; all_trade_records.append(rec)
                        last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue
                        
                # IDLE Entries
                if pos is None and idx >= 15:
                    tact = bar.get("tactical_probs", [0.33, 0.33, 0.33])
                    meso = bar.get("meso_probs", [0.33, 0.33, 0.33])
                    micro = bar.get("micro_probs", [0.33, 0.33, 0.33])
                    macro = bar.get("macro_probs", [0.33, 0.33, 0.33])
                    
                    d_tact = tact[2] - tact[0]
                    d_micro = micro[2] - micro[0]
                    d_meso = meso[2] - meso[0]
                    d_macro = macro[2] - macro[0]
                    
                    is_long = (d_tact >= 0.08 and d_meso >= 0.015 and d_micro >= 0.01 and d_macro >= -0.02)
                    is_short = (d_tact <= -0.08 and d_meso <= -0.015 and d_micro <= -0.01 and d_macro <= 0.02)
                    
                    cand_side = None
                    if is_long: cand_side = "BUY"
                    elif is_short: cand_side = "SELL"
                    
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
                            
                        pos = {"side": cand_side, "entry_px": ask if cand_side == "BUY" else bid, "ts": ts_ms, "mfe": 0.0, "mae": 0.0, "max_drop_from_peak": 0.0}
                        
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
        results.append({
            "cfg": cfg,
            "trades": total_trades,
            "wins": total_wins,
            "losses": total_trades - total_wins,
            "wr": wr,
            "total_net": total_net,
            "avg_net": avg,
            "wr100": wr100_cycles,
            "active": active_cycles,
            "profitable": profitable_cycles,
            "tier_counts": dict(tier_counts),
            "records": all_trade_records
        })
        
    print("="*95)
    print("COMPARATIVE RE-ANALYSIS: RESILIENCE AGAINST SUDDEN DROPS ACROSS 64 CYCLES")
    print("="*95)
    print(f"{'Stop Loss Configuration':<32} | {'Trades':<6} | {'Win Rate':<8} | {'Net Realized PnL':<16} | {'Avg/Trade':<9} | {'100% WR Cycles':<14}")
    print("-" * 95)
    for r in results:
        print(f"{r['cfg']['name']:<32} | {r['trades']:<6} | {r['wr']:>6.1f}%  | {r['total_net']:>+12.2f} bps | {r['avg_net']:>+7.2f}b | {r['wr100']}/{r['active']} ({r['wr100']/max(1,r['active'])*100:.1f}%)")
    print("="*95)
    
    # Detailed analysis of drops on the Taker -45b configuration
    r_taker45 = [r for r in results if r['cfg']['name'] == 'Taker -45.0 bps (Aggressive Emergency Stop)'][0]
    trades = r_taker45['records']
    maes = [t['mae'] for t in trades]
    sl_trades = [t for t in trades if t['reason'] == 'STOP_LOSS']
    
    print("\n[FORENSIC AUDIT OF SUDDEN DROPS]")
    print(f"Total Trades Evaluated: {len(trades)}")
    print(f"Median Drop (MAE) on all trades: {sorted(maes)[len(maes)//2]:.2f} bps")
    print(f"Worst Drop (MAE) in entire 64 cycles: {min(maes):.2f} bps")
    print(f"\nBreakdown of Drops Triggering Stop Loss ({len(sl_trades)} occurrences):")
    for st in sl_trades:
        print(f"  Cycle: {st['cycle']:<18} Side: {st['side']} Net PnL: {st['net']:>7.2f}b Dur: {st['dur']:>6.1f}s MAE: {st['mae']:>7.2f}b DropFromPeak: {st['max_drop']:>7.2f}b ActualGross: {st['actual_gross']:>7.2f}b")

    # Let's inspect the distribution of drops survived without stopping out
    survived_drops = [t for t in trades if t['reason'] != 'STOP_LOSS']
    survived_maes = [t['mae'] for t in survived_drops]
    print(f"\nDrops Successfully Survived & Closed with Profit ({len(survived_drops)} trades):")
    print(f"  Max Drop Survived & Ended Profitable: {min(survived_maes):.2f} bps")
    print(f"  Number of Winning Trades with MAE between -10b and -30b: {sum(1 for m in survived_maes if -30.0 <= m <= -10.0)}")
    print(f"  Number of Winning Trades with MAE between -30b and -44b: {sum(1 for m in survived_maes if -44.0 <= m < -30.0)}")

if __name__ == "__main__":
    analyze_drops_and_stop_loss()
