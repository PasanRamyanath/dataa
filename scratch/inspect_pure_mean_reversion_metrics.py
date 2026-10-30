import os
import glob
import json

base_dir = "D:/PrivateProjects/BTC Futures/scratch"
archives = sorted([d for d in os.listdir(base_dir) if (d.startswith("archive_cycle") or d.startswith("archive_v3")) and os.path.isdir(os.path.join(base_dir, d))])

tp_target_bps = 35.0
time_limit_sec = 900.0

all_trades = []

for arch in archives:
    ledgers = glob.glob(os.path.join(base_dir, arch, "*ledger*.jsonl"))
    if not ledgers: continue
    with open(ledgers[0], "r", encoding="utf-8") as f:
        bars = [json.loads(l) for l in f if l.strip().startswith("{")]
        
    pos = None
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
            
            floor = None
            reason = None
            if mfe >= 10.0: 
                floor = 8.5
                reason = "TIER2_RUNNER_RATCHET_TP"
            elif mfe >= 6.0: 
                floor = 6.25
                reason = "TIER1_STEPPED_RATCHET_TP"
            elif mfe >= 3.8: 
                floor = 4.15
                reason = "TIER0_BREAKEVEN_MAKER_SHIELD"
            
            if gross >= tp_target_bps:
                net = gross - 4.0
                all_trades.append({"arch": arch, "side": side, "net": net, "dur": dur, "mfe": mfe, "mae": pos["mae"], "reason": "TARGET_MAKER_TP"})
                last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue
            elif floor is not None and gross <= floor:
                net = floor - 4.0
                all_trades.append({"arch": arch, "side": side, "net": net, "dur": dur, "mfe": mfe, "mae": pos["mae"], "reason": reason})
                last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue
            elif dur >= time_limit_sec and gross >= 4.0:
                net = gross - 4.0
                all_trades.append({"arch": arch, "side": side, "net": net, "dur": dur, "mfe": mfe, "mae": pos["mae"], "reason": "TIME_UNWIND"})
                last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue
                
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

print(f"Total Closed Trades across all historical archives: {len(all_trades)}")
wins = [t for t in all_trades if t['net'] > 0]
losses = [t for t in all_trades if t['net'] < 0]
be = [t for t in all_trades if t['net'] == 0]
print(f"Wins: {len(wins)} | Losses: {len(losses)} | Break-Even: {len(be)} | Win Rate: {len(wins)/len(all_trades)*100.0:.2f}%")

total_net = sum([t['net'] for t in all_trades])
avg_net = total_net / len(all_trades)
print(f"Total Cumulative Realized Net Cash Edge: {total_net:+.2f} bps")
print(f"Average Profit per Trade: {avg_net:+.2f} bps/trade")

# Breakdown by exit reason
reasons = {}
for t in all_trades:
    r = t['reason']
    if r not in reasons: reasons[r] = []
    reasons[r].append(t)

print("\n--- Distribution of Trades by Exit Reason ---")
for r, tlist in sorted(reasons.items(), key=lambda x: len(x[1]), reverse=True):
    sub_net = sum([t['net'] for t in tlist])
    avg_sub = sub_net / len(tlist)
    avg_dur = sum([t['dur'] for t in tlist]) / len(tlist)
    avg_mae = sum([t['mae'] for t in tlist]) / len(tlist)
    print(f"  {r:<30}: {len(tlist):<4} trades | Net: {sub_net:>+8.2f} bps | Avg: {avg_sub:>+5.2f} bps/tr | Avg Dur: {avg_dur:>5.1f}s | Avg MAE: {avg_mae:>5.2f}b")

# Maximum MAE observed across winning trades
maes = [t['mae'] for t in all_trades]
print(f"\nMAE Distribution across all 219 winning trades:")
print(f"  Median MAE: {sorted(maes)[len(maes)//2]:.2f} bps")
print(f"  75th percentile MAE: {sorted(maes)[int(len(maes)*0.75)]:.2f} bps")
print(f"  Worst MAE absorbed before winning: {min(maes):.2f} bps")
