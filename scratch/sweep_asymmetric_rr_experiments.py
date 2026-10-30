import os
import glob
import json
import collections

base_dir = "D:/PrivateProjects/BTC Futures/scratch"
archives = sorted([d for d in os.listdir(base_dir) if (d.startswith("archive_cycle") or d.startswith("archive_v3")) and os.path.isdir(os.path.join(base_dir, d))])

def run_experiment(buffer_bps, tp_target_bps, time_limit_sec, name):
    total_trades = 0
    total_wins = 0
    total_net = 0.0
    profitable_cycles = 0
    wr100_cycles = 0
    active_cycles = 0
    
    for arch in archives:
        ledgers = glob.glob(os.path.join(base_dir, arch, "*ledger*.jsonl"))
        if not ledgers: continue
        with open(ledgers[0], "r", encoding="utf-8") as f:
            bars = [json.loads(l) for l in f if l.strip().startswith("{")]
            
        pos = None
        closed = []
        cum_pnl = 0.0
        last_exit_ts = 0
        last_exit_side = ""
        last_exit_px = 0.0
        
        for idx, bar in enumerate(bars):
            mid = bar.get("bbo_mid", bar.get("mid", 0.0))
            bid = bar.get("bbo_bid", mid - 0.05)
            ask = bar.get("bbo_ask", mid + 0.05)
            ts_ms = bar.get("ts_ms", idx * 10000)
            
            if pos is not None:
                side = pos["side"]
                epx = pos["entry_px"]
                gross = (bid - epx)/epx*10000.0 if side == "BUY" else (epx - ask)/epx*10000.0
                pos["mfe"] = max(pos["mfe"], gross)
                pos["mae"] = min(pos["mae"], gross)
                mfe = pos["mfe"]
                dur = (ts_ms - pos["ts"]) / 1000.0
                
                # Shield & Ratchets
                floor = None
                if mfe >= 10.0: floor = 8.5
                elif mfe >= 6.0: floor = 6.25
                elif mfe >= 3.8: floor = 4.15
                
                if gross >= tp_target_bps:
                    net = gross - 4.0
                    closed.append(net); cum_pnl += net; last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue
                elif floor is not None and gross <= floor:
                    net = floor - 4.0
                    closed.append(net); cum_pnl += net; last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue
                elif dur >= time_limit_sec and gross >= 4.0:
                    net = gross - 4.0
                    closed.append(net); cum_pnl += net; last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue
                elif buffer_bps is not None and gross <= -buffer_bps:
                    net = -buffer_bps - 4.0
                    closed.append(net); cum_pnl += net; last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue
                    
            if pos is None and idx >= 10:
                tact = bar.get("tactical_probs", [0.33, 0.33, 0.33])
                meso = bar.get("meso_probs", [0.33, 0.33, 0.33])
                micro = bar.get("micro_probs", [0.33, 0.33, 0.33])
                p_dir = bar.get("primary_direction", bar.get("direction", 1))
                edge = bar.get("expected_net_edge_bps", bar.get("edge", 0.0))
                
                d_tact = tact[2] - tact[0]
                d_micro = micro[2] - micro[0]
                d_meso = meso[2] - meso[0]
                
                cand_dir = None
                tact_dir = 2 if d_tact >= 0.14 else (0 if d_tact <= -0.14 else 1)
                tact_confirmed = (tact_dir in (0, 2)) and (
                    (tact_dir == 2 and d_micro > 0.02 and d_meso >= 0.00) or
                    (tact_dir == 0 and d_micro < -0.02 and d_meso <= 0.00)
                )
                
                if tact_confirmed: cand_dir = tact_dir
                elif p_dir in (0, 2) and (edge >= 28.0 or (edge >= 4.0 and edge <= 20.0)):
                    if p_dir == 2 and d_meso >= 0.00: cand_dir = 2
                    elif p_dir == 0 and d_meso <= 0.00: cand_dir = 0
                    
                if cand_dir in (0, 2):
                    if last_exit_ts > 0 and (ts_ms - last_exit_ts) / 1000.0 < 300.0 and last_exit_px > 0:
                        if cand_dir == 2 and last_exit_side == "BUY" and mid > (last_exit_px - 15.0): continue
                        if cand_dir == 0 and last_exit_side == "SELL" and mid < (last_exit_px + 15.0): continue
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
    print(f"{name:<35} | Tr: {total_trades:<5} | WR: {wr:>5.1f}% | Net PnL: {total_net:>+9.2f} bps | Avg/Tr: {avg:>+5.2f} bps | 100% WR Cycles: {wr100_cycles}/{active_cycles} | Profitable Cycles: {profitable_cycles}/{active_cycles}")

print("="*115)
print("SYSTEMATIC ASYMMETRIC R:R EXPERIMENTAL SWEEP ACROSS ALL 63 HISTORICAL CYCLES")
print("="*115)

# Sweep configurations
run_experiment(buffer_bps=18.0, tp_target_bps=35.0, time_limit_sec=450.0, name="Baseline (Buffer 18b, TP 35b)")
run_experiment(buffer_bps=10.0, tp_target_bps=35.0, time_limit_sec=450.0, name="Tight Cut (Buffer 10b, TP 35b)")
run_experiment(buffer_bps=6.5, tp_target_bps=35.0, time_limit_sec=450.0, name="Ultra-Tight Cut (Buffer 6.5b, TP 35b)")
run_experiment(buffer_bps=25.0, tp_target_bps=35.0, time_limit_sec=600.0, name="Wide Breathe (Buffer 25b, TP 35b)")
run_experiment(buffer_bps=None, tp_target_bps=35.0, time_limit_sec=900.0, name="V3.5 Pure Mean-Reversion (No SL Cut)")
