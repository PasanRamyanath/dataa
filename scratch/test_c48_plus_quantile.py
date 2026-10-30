import os
import glob
import json
import collections
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

base_dir = "D:/PrivateProjects/BTC Futures/scratch"
archives = sorted([d for d in os.listdir(base_dir) if (d.startswith("archive_cycle") or d.startswith("archive_v3")) and os.path.isdir(os.path.join(base_dir, d))])

def run_cycle48_plus_quantile():
    maker_fee = 2.0
    total_trades = 0
    total_wins = 0
    total_net = 0.0
    active_cycles = 0
    wr100_cycles = 0
    profitable_cycles = 0
    
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
            
            # 1. Manage Active Position
            if pos is not None:
                side = pos["side"]; epx = pos["entry_px"]
                gross = (bid - epx)/epx*10000.0 if side == "BUY" else (epx - ask)/epx*10000.0
                pos["mfe"] = max(pos["mfe"], gross)
                pos["mae"] = min(pos["mae"], gross)
                mfe = pos["mfe"]; dur = (ts_ms - pos["ts"]) / 1000.0
                
                # Cycle 48 Exact Dual-Tier Stepped Ratchet Ladder
                floor_gross = None
                if mfe >= 15.0: floor_gross = 12.50
                elif mfe >= 10.0: floor_gross = 8.50
                elif mfe >= 6.0: floor_gross = 6.00
                elif mfe >= 4.5: floor_gross = 5.25 # Tier 0.5 ratchet
                elif mfe >= 3.8: floor_gross = 4.15 # Shield
                    
                if gross >= 35.0:
                    net = gross - 4.0; closed.append(net); cum_pnl += net
                    last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue
                elif floor_gross is not None and gross <= floor_gross:
                    net = floor_gross - 4.0; closed.append(net); cum_pnl += net
                    last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue
                elif dur >= 900.0 and gross >= 4.0:
                    net = gross - 4.0; closed.append(net); cum_pnl += net
                    last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue
                elif gross <= -45.0: # Circuit breaker
                    net = -45.0 - 4.0; closed.append(net); cum_pnl += net
                    last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue
                    
            # 2. Evaluate IDLE Entries with Cycle 48 Quad Concordance + Quantile Guard
            if pos is None and idx >= 15:
                tact = bar.get("tactical_probs", [0.33, 0.33, 0.33])
                meso = bar.get("meso_probs", [0.33, 0.33, 0.33])
                micro = bar.get("micro_probs", [0.33, 0.33, 0.33])
                macro = bar.get("macro_probs", [0.33, 0.33, 0.33])
                
                d_tact = tact[2] - tact[0]
                d_micro = micro[2] - micro[0]
                d_meso = meso[2] - meso[0]
                d_macro = macro[2] - macro[0]
                
                is_long = (d_tact > 0.05 and d_meso > 0.01 and d_micro > 0.01 and d_macro >= 0.0)
                is_short = (d_tact < -0.05 and d_meso < -0.01 and d_micro < -0.01 and d_macro <= 0.0)
                
                cand_side = None
                if is_long: cand_side = "BUY"
                elif is_short: cand_side = "SELL"
                
                if cand_side is not None:
                    # Quantile Range Guard
                    if len(rolling_15m) >= 30:
                        hi = max(rolling_15m); lo = min(rolling_15m); rng = hi - lo
                        if rng >= 20.0:
                            pos_in_rng = (mid - lo) / rng
                            if cand_side == "BUY" and pos_in_rng >= 0.75: continue
                            if cand_side == "SELL" and pos_in_rng <= 0.25: continue
                            
                    # Retrace Barrier
                    if last_exit_ts > 0 and (ts_ms - last_exit_ts) / 1000.0 < 200.0:
                        if cand_side == "BUY" and last_exit_side == "BUY" and mid > (last_exit_px - 10.0): continue
                        if cand_side == "SELL" and last_exit_side == "SELL" and mid < (last_exit_px + 10.0): continue
                        
                    pos = {"side": cand_side, "entry_px": ask if cand_side == "BUY" else bid, "ts": ts_ms, "mfe": 0.0, "mae": 0.0}
                    
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
    print(f"==========================================================================")
    print(f"CYCLE 48 DUAL-TIER RATCHET + QUANTILE RANGE GUARD (ALL-TIME REPLAY)")
    print(f"==========================================================================")
    print(f"Total Trades: {total_trades} across {active_cycles} active sessions ({total_trades/max(1,active_cycles):.2f} tr/session)")
    print(f"Wins: {total_wins} | Losses: {total_trades - total_wins}")
    print(f"Win Rate: {wr:.1f}%")
    print(f"Total Net Realized PnL: {total_net:+.2f} bps")
    print(f"Average Profit per Trade: {avg:+.2f} bps")
    print(f"Sessions with 100% Win Rate: {wr100_cycles} / {active_cycles} ({wr100_cycles/max(1,active_cycles)*100.0:.1f}%)")
    print(f"Sessions strictly Net Profitable: {profitable_cycles} / {active_cycles} ({profitable_cycles/max(1,active_cycles)*100.0:.1f}%)")
    print(f"==========================================================================")

run_cycle48_plus_quantile()
