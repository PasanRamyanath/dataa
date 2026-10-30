import os
import glob
import json
import collections
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

base_dir = "D:/PrivateProjects/BTC Futures/scratch"
archives = sorted([d for d in os.listdir(base_dir) if (d.startswith("archive_cycle") or d.startswith("archive_v3")) and os.path.isdir(os.path.join(base_dir, d))])

def diagnose_losses():
    loss_trades = []
    
    for arch in archives:
        ledgers = glob.glob(os.path.join(base_dir, arch, "*ledger*.jsonl"))
        if not ledgers: continue
        with open(ledgers[0], "r", encoding="utf-8") as f:
            bars = [json.loads(l) for l in f if l.strip().startswith("{")]
            
        rolling_15m = collections.deque(maxlen=90)
        pos = None
        last_exit_ts = 0; last_exit_side = ""; last_exit_px = 0.0
        
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
                
                floor_gross = None
                if mfe >= 15.0: floor_gross = 12.50
                elif mfe >= 10.0: floor_gross = 8.50
                elif mfe >= 6.0: floor_gross = 6.25
                elif mfe >= 4.5: floor_gross = 5.25
                elif mfe >= 3.8: floor_gross = 4.15
                
                if gross >= 35.0:
                    last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue
                elif floor_gross is not None and gross <= floor_gross:
                    last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue
                elif dur >= 900.0 and gross >= 4.0:
                    last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue
                elif gross <= -45.0:
                    pos["exit_bar"] = idx
                    pos["final_gross"] = gross
                    pos["dur"] = dur
                    loss_trades.append(pos)
                    last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue

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
                
                cand_side = "BUY" if is_long else ("SELL" if is_short else None)
                if cand_side is not None:
                    pos_in_rng = 0.5
                    if len(rolling_15m) >= 30:
                        hi = max(rolling_15m); lo = min(rolling_15m); rng = hi - lo
                        if rng >= 20.0:
                            pos_in_rng = (mid - lo) / rng
                            if cand_side == "BUY" and pos_in_rng >= 0.80: continue
                            if cand_side == "SELL" and pos_in_rng <= 0.20: continue
                    if last_exit_ts > 0 and (ts_ms - last_exit_ts) / 1000.0 < 200.0:
                        if cand_side == "BUY" and last_exit_side == "BUY" and mid > (last_exit_px - 10.0): continue
                        if cand_side == "SELL" and last_exit_side == "SELL" and mid < (last_exit_px + 10.0): continue
                        
                    pos = {
                        "cycle": arch,
                        "side": cand_side,
                        "entry_bar": idx,
                        "entry_px": ask if cand_side == "BUY" else bid,
                        "entry_mid": mid,
                        "ts": ts_ms,
                        "pos_in_rng": pos_in_rng,
                        "d_tact": d_tact,
                        "d_meso": d_meso,
                        "d_macro": d_macro,
                        "d_micro": d_micro,
                        "edge": bar.get("expected_net_edge_bps", bar.get("edge", 0.0)),
                        "p_dir": bar.get("primary_direction", bar.get("direction", 1)),
                        "mfe": 0.0,
                        "mae": 0.0
                    }
                    
    print(f"Total Losses Found: {len(loss_trades)}")
    for i, lt in enumerate(loss_trades):
        print(f"\n--- Loss {i+1}: Cycle {lt['cycle']} ---")
        print(f"  Side: {lt['side']} at Bar {lt['entry_bar']} -> Exited at Bar {lt['exit_bar']} (Dur: {lt['dur']:.1f}s)")
        print(f"  Entry Px: {lt['entry_px']:.2f} | Pos in 15m Rng: {lt['pos_in_rng']:.2f}")
        print(f"  Model Deltas: Tact={lt['d_tact']:+.3f}, Meso={lt['d_meso']:+.3f}, Macro={lt['d_macro']:+.3f}, Micro={lt['d_micro']:+.3f}")
        print(f"  Primary Direction: {lt['p_dir']} | Expected Edge: {lt['edge']:.2f} bps")
        print(f"  Peak MFE: {lt['mfe']:.2f} bps | Worst MAE: {lt['mae']:.2f} bps")

if __name__ == "__main__":
    diagnose_losses()
