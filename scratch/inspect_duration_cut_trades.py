import os
import glob
import json
import collections
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

base_dir = "D:/PrivateProjects/BTC Futures/scratch"
archives = sorted([d for d in os.listdir(base_dir) if (d.startswith("archive_cycle") or d.startswith("archive_v3")) and os.path.isdir(os.path.join(base_dir, d))])

def inspect_duration_cuts():
    # Let's inspect the trades that triggered HARD_DURATION_CUT in Configuration 2
    # Ladder: [(35.0, 35.0, "TP"), (20.0, 16.00, "TIER4_EXP"), (14.0, 11.00, "TIER3_RUN"), (9.0, 7.00, "TIER2_MID"), (6.5, 5.25, "TIER1_C48"), (5.0, 4.15, "TIER0_BE")]
    maker_fee = 2.0
    taker_fee = 4.5
    
    ladder = [(35.0, 35.0, "TP"), (20.0, 16.00, "TIER4_EXP"), (14.0, 11.00, "TIER3_RUN"), (9.0, 7.00, "TIER2_MID"), (6.5, 5.25, "TIER1_C48"), (5.0, 4.15, "TIER0_BE")]
    
    cut_trades = []
    
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
                
                if gross >= 35.0:
                    last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue
                    
                floor_gross = None
                for hurdle, floor, name in ladder:
                    if name == "TP": continue
                    if mfe >= hurdle:
                        floor_gross = floor
                        break
                        
                if floor_gross is not None and gross <= floor_gross:
                    last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue
                elif dur >= 900.0 and gross >= 4.0:
                    last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue
                elif dur >= 1500.0:
                    net = gross - 4.0
                    cut_trades.append({"cycle": arch, "side": side, "entry_bar": pos["bar"], "exit_bar": idx, "net": net, "gross": gross, "mfe": mfe, "mae": pos["mae"]})
                    last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue
                elif gross <= -45.0:
                    last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue

            if pos is None and idx >= 15 and idx <= 270:
                p_dir = bar.get("primary_direction", bar.get("direction", 1))
                edge = bar.get("expected_net_edge_bps", bar.get("edge", 0.0))
                if p_dir == 1 or edge < 8.0: continue
                
                tact = bar.get("tactical_probs", [0.33, 0.33, 0.33])
                meso = bar.get("meso_probs", [0.33, 0.33, 0.33])
                d_tact = tact[2] - tact[0]
                d_meso = meso[2] - meso[0]
                
                cand_side = None
                if p_dir == 2 and d_tact >= 0.05 and d_meso >= 0.01: cand_side = "BUY"
                elif p_dir == 0 and d_tact <= -0.05 and d_meso <= -0.01: cand_side = "SELL"
                
                if cand_side is not None:
                    if len(rolling_15m) >= 30:
                        hi = max(rolling_15m); lo = min(rolling_15m); rng = hi - lo
                        if rng >= 20.0:
                            pos_in_rng = (mid - lo) / rng
                            if cand_side == "BUY" and pos_in_rng >= 0.75: continue
                            if cand_side == "SELL" and pos_in_rng <= 0.25: continue
                    if last_exit_ts > 0 and (ts_ms - last_exit_ts) / 1000.0 < 200.0:
                        if cand_side == "BUY" and last_exit_side == "BUY" and mid > (last_exit_px - 10.0): continue
                        if cand_side == "SELL" and last_exit_side == "SELL" and mid < (last_exit_px + 10.0): continue
                    pos = {"side": cand_side, "entry_px": ask if cand_side == "BUY" else bid, "bar": idx, "ts": ts_ms, "mfe": 0.0, "mae": 0.0}

    print("="*95)
    print(f"ANATOMY OF THE {len(cut_trades)} TRADES CUT AT 25 MINUTES (1500s)")
    print("="*95)
    print(f"{'Cycle':<18} | {'Side':<4} | {'Entry':<6} | {'Exit':<6} | {'Gross PnL':<10} | {'Net PnL':<10} | {'MFE':<8} | {'MAE':<8}")
    print("-" * 95)
    for c in cut_trades:
        print(f"{c['cycle']:<18} | {c['side']:<4} | Bar {c['entry_bar']:<2} | Bar {c['exit_bar']:<2} | {c['gross']:>+8.2f}b | {c['net']:>+8.2f}b | {c['mfe']:>+6.2f}b | {c['mae']:>+6.2f}b")
    print("="*95)
    print(f"Total Net PnL of 25m Cuts: {sum(c['net'] for c in cut_trades):+.2f} bps")

if __name__ == "__main__":
    inspect_duration_cuts()
