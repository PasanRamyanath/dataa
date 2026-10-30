import os
import glob
import json
import collections

base_dir = "D:/PrivateProjects/BTC Futures/scratch"
archives = sorted([d for d in os.listdir(base_dir) if (d.startswith("archive_cycle") or d.startswith("archive_v3")) and os.path.isdir(os.path.join(base_dir, d))])

from test_certified_robust_peak_engine import RobustPeakArchitecture

engine = RobustPeakArchitecture(catastrophic_breaker=35.0, tp_target=35.0)

losses = []

for arch in archives:
    ledgers = glob.glob(os.path.join(base_dir, arch, "*ledger*.jsonl"))
    if not ledgers: continue
    
    # Run engine logic and record losses
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
        
        if pos is not None:
            side = pos["side"]; epx = pos["entry_px"]
            gross = (bid - epx)/epx*10000.0 if side == "BUY" else (epx - ask)/epx*10000.0
            pos["mfe"] = max(pos["mfe"], gross)
            pos["mae"] = min(pos["mae"], gross)
            mfe = pos["mfe"]; dur = (ts_ms - pos["ts"]) / 1000.0
            
            floor = None; reason = None
            if mfe >= 10.0: floor = 8.5; reason = "TIER2_RUNNER_RATCHET_TP"
            elif mfe >= 6.0: floor = 6.25; reason = "TIER1_STEPPED_RATCHET_TP"
            elif mfe >= 3.8: floor = 4.15; reason = "TIER0_BREAKEVEN_MAKER_SHIELD"
            
            if gross >= 35.0:
                pos = None; last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; last_exit_was_win = True
            elif floor is not None and gross <= floor:
                pos = None; last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; last_exit_was_win = True
            elif dur >= 900.0 and gross >= 4.0:
                pos = None; last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; last_exit_was_win = True
            elif gross <= -35.0:
                losses.append({
                    "cycle": arch,
                    "side": side,
                    "bar_idx": idx,
                    "dur": dur,
                    "mfe": mfe,
                    "mae": pos["mae"],
                    "entry_px": epx,
                    "exit_px": ask if side == "BUY" else bid,
                    "net_loss": -35.0 - 4.0
                })
                pos = None; last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; last_exit_was_win = False
                
        if pos is None and idx >= 15:
            tact = bar.get("tactical_probs", [0.33, 0.33, 0.33])
            meso = bar.get("meso_probs", [0.33, 0.33, 0.33])
            micro = bar.get("micro_probs", [0.33, 0.33, 0.33])
            p_dir = bar.get("primary_direction", bar.get("direction", 1))
            edge = bar.get("expected_net_edge_bps", bar.get("edge", 0.0))
            
            conf_tact = max(tact); conf_meso = max(meso)
            d_tact = tact[2] - tact[0]; d_micro = micro[2] - micro[0]; d_meso = meso[2] - meso[0]
            
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

print(f"Total catastrophic losses: {len(losses)}")
print("\nForensic Analysis of Losses:")
for l in losses:
    print(f"  {l['cycle']:<20} | Side: {l['side']} | Bar: {l['bar_idx']:<3} | Dur: {l['dur']:>5.0f}s | Peak MFE: {l['mfe']:>5.2f}b | MAE: {l['mae']:>6.2f}b")
