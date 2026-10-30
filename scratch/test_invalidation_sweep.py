import os
import glob
import json
import collections

base_dir = "D:/PrivateProjects/BTC Futures/scratch"
archives = sorted([d for d in os.listdir(base_dir) if (d.startswith("archive_cycle") or d.startswith("archive_v3")) and os.path.isdir(os.path.join(base_dir, d))])

def test_peak_engine(model_inval=True, time_stagnation_sec=450, maker_inval=True):
    total_trades = 0
    total_wins = 0
    total_net = 0.0
    active_cycles = 0
    wr100_cycles = 0
    profitable_cycles = 0
    exit_reasons = collections.Counter()
    
    maker_fee = 2.0
    taker_fee = 5.0
    
    for arch in archives:
        ledgers = glob.glob(os.path.join(base_dir, arch, "*ledger*.jsonl"))
        if not ledgers: continue
        with open(ledgers[0], "r", encoding="utf-8") as f:
            bars = [json.loads(l) for l in f if l.strip().startswith("{")]
            
        rolling_15m = collections.deque(maxlen=90)
        pos = None
        closed = []
        cum_pnl = 0.0
        last_exit_ts = 0; last_exit_side = ""; last_exit_px = 0.0; last_exit_was_win = True
        
        for idx, bar in enumerate(bars):
            mid = bar.get("bbo_mid", bar.get("mid", 0.0))
            bid = bar.get("bbo_bid", mid - 0.05)
            ask = bar.get("bbo_ask", mid + 0.05)
            ts_ms = bar.get("ts_ms", idx * 10000)
            rolling_15m.append(mid)
            
            tact = bar.get("tactical_probs", [0.33, 0.33, 0.33])
            meso = bar.get("meso_probs", [0.33, 0.33, 0.33])
            micro = bar.get("micro_probs", [0.33, 0.33, 0.33])
            d_tact = tact[2] - tact[0]
            d_meso = meso[2] - meso[0]
            
            if pos is not None:
                side = pos["side"]; epx = pos["entry_px"]
                gross = (bid - epx)/epx*10000.0 if side == "BUY" else (epx - ask)/epx*10000.0
                pos["mfe"] = max(pos["mfe"], gross)
                pos["mae"] = min(pos["mae"], gross)
                mfe = pos["mfe"]; dur = (ts_ms - pos["ts"]) / 1000.0
                
                floor = None; reason = None
                if mfe >= 10.0: floor = 8.50; reason = "TIER2_RUNNER_RATCHET_TP"
                elif mfe >= 6.0: floor = 6.25; reason = "TIER1_STEPPED_RATCHET_TP"
                elif mfe >= 3.8: floor = 4.15; reason = "TIER0_BREAKEVEN_MAKER_SHIELD"
                
                # Check target TP
                if gross >= 35.0:
                    net = gross - (2 * maker_fee)
                    closed.append(net); cum_pnl += net; exit_reasons["TARGET_TP"] += 1
                    last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; last_exit_was_win = True; pos = None; continue
                # Check Ratchet / Shield floor
                elif floor is not None and gross <= floor:
                    net = floor - (2 * maker_fee)
                    closed.append(net); cum_pnl += net; exit_reasons[reason] += 1
                    last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; last_exit_was_win = True; pos = None; continue
                # Check profitable unwind
                elif dur >= 900.0 and gross >= 4.0:
                    net = gross - (2 * maker_fee)
                    closed.append(net); cum_pnl += net; exit_reasons["PROFIT_UNWIND"] += 1
                    last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; last_exit_was_win = True; pos = None; continue
                
                # Check Model Invalidation (Meso & Tactical flip opposite)
                inval = False
                if model_inval:
                    if side == "BUY" and d_meso <= -0.06 and d_tact <= -0.10:
                        inval = True
                    elif side == "SELL" and d_meso >= 0.06 and d_tact >= 0.10:
                        inval = True
                        
                if inval and mfe < 3.8:
                    fee = (2 * maker_fee) if maker_inval else (maker_fee + taker_fee)
                    net = gross - fee
                    closed.append(net); cum_pnl += net; exit_reasons["MODEL_INVAL"] += 1
                    last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; last_exit_was_win = (net > 0); pos = None; continue
                    
                # Time Stagnation (failed to reach shield within X seconds and currently underwater)
                if time_stagnation_sec is not None and dur >= time_stagnation_sec and mfe < 3.0 and gross < 0.0:
                    fee = (2 * maker_fee) if maker_inval else (maker_fee + taker_fee)
                    net = gross - fee
                    closed.append(net); cum_pnl += net; exit_reasons["TIME_STAGNATION"] += 1
                    last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; last_exit_was_win = (net > 0); pos = None; continue
                    
                # Catastrophic structural stop at -35 bps
                if gross <= -35.0:
                    net = -35.0 - (2 * maker_fee)
                    closed.append(net); cum_pnl += net; exit_reasons["CATASTROPHIC_BREAKER"] += 1
                    last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; last_exit_was_win = False; pos = None; continue
                    
            # Entries
            if pos is None and idx >= 15:
                p_dir = bar.get("primary_direction", bar.get("direction", 1))
                edge = bar.get("expected_net_edge_bps", bar.get("edge", 0.0))
                micro = bar.get("micro_probs", [0.33, 0.33, 0.33])
                d_micro = micro[2] - micro[0]
                
                conf_tact = max(tact); conf_meso = max(meso)
                has_conviction = (conf_tact >= 0.42 and conf_meso >= 0.38) or (abs(d_tact) >= 0.14 and abs(d_meso) >= 0.03)
                if not has_conviction: continue
                
                cand_dir = None
                tact_dir = 2 if d_tact >= 0.12 else (0 if d_tact <= -0.12 else 1)
                tact_confirmed = (tact_dir in (0, 2)) and (
                    (tact_dir == 2 and d_micro > 0.012 and d_meso >= 0.02) or
                    (tact_dir == 0 and d_micro < -0.012 and d_meso <= -0.02)
                )
                if tact_confirmed: cand_dir = tact_dir
                elif p_dir in (0, 2) and (edge >= 28.0 or (edge >= 5.0 and edge <= 20.0)):
                    if p_dir == 2 and d_meso >= 0.02: cand_dir = 2
                    elif p_dir == 0 and d_meso <= -0.02: cand_dir = 0
                    
                if cand_dir in (0, 2):
                    if len(rolling_15m) >= 30:
                        hi = max(rolling_15m); lo = min(rolling_15m); rng = hi - lo
                        if rng >= 25.0:
                            if cand_dir == 2 and mid >= hi - 6.0: continue
                            if cand_dir == 0 and mid <= lo + 6.0: continue
                    if last_exit_ts > 0:
                        elapsed = (ts_ms - last_exit_ts) / 1000.0
                        if not last_exit_was_win and elapsed < 120.0: continue
                        elif last_exit_was_win and elapsed < 15.0: continue
                        if elapsed < 200.0 and last_exit_px > 0:
                            if cand_dir == 2 and last_exit_side == "BUY" and mid > (last_exit_px - 10.0): continue
                            if cand_dir == 0 and last_exit_side == "SELL" and mid < (last_exit_px + 10.0): continue
                    pos = {"side": "BUY" if cand_dir == 2 else "SELL", "entry_px": ask if cand_dir == 2 else bid, "ts": ts_ms, "mfe": 0.0, "mae": 0.0}
                    
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
    return {
        "trades": total_trades,
        "wins": total_wins,
        "wr": wr,
        "net_pnl": total_net,
        "avg": avg,
        "active_cycles": active_cycles,
        "wr100_cycles": wr100_cycles,
        "profitable_cycles": profitable_cycles,
        "reasons": dict(exit_reasons)
    }

print("Running parameter grid across all 64 cycles...")
for stag in [None, 300, 450, 600]:
    for model_inv in [False, True]:
        res = test_peak_engine(model_inval=model_inv, time_stagnation_sec=stag, maker_inval=True)
        print(f"Stag: {str(stag):<4} | ModelInv: {str(model_inv):<5} -> Tr: {res['trades']:<3} | WR: {res['wr']:>5.1f}% | Net: {res['net_pnl']:>8.2f} bps | Avg: {res['avg']:>5.2f} bps | 100% WR Cyc: {res['wr100_cycles']}/{res['active_cycles']} | Profitable: {res['profitable_cycles']}/{res['active_cycles']}")
        print(f"   Exits: {res['reasons']}")
