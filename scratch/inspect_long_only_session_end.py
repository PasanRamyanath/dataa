import os
import glob
import json
import collections
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

base_dir = "D:/PrivateProjects/BTC Futures/scratch"
archives = sorted([d for d in os.listdir(base_dir) if (d.startswith("archive_cycle") or d.startswith("archive_v3")) and os.path.isdir(os.path.join(base_dir, d))])

def inspect_long_only_session_end():
    # Inspect the 12 session-end trades from Experiment 8
    maker_fee = 2.0
    taker_fee = 4.5
    
    session_end_trades = []
    
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
                epx = pos["entry_px"]
                gross = (bid - epx)/epx*10000.0
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
                    last_exit_ts = ts_ms; last_exit_side = "BUY"; last_exit_px = mid; pos = None; continue
                elif floor_gross is not None and gross <= floor_gross:
                    last_exit_ts = ts_ms; last_exit_side = "BUY"; last_exit_px = mid; pos = None; continue
                elif dur >= 900.0 and gross >= 4.0:
                    last_exit_ts = ts_ms; last_exit_side = "BUY"; last_exit_px = mid; pos = None; continue
                elif gross <= -45.0:
                    last_exit_ts = ts_ms; last_exit_side = "BUY"; last_exit_px = mid; pos = None; continue

            if pos is None and idx >= 15 and idx <= 270:
                p_dir = bar.get("primary_direction", bar.get("direction", 1))
                edge = bar.get("expected_net_edge_bps", bar.get("edge", 0.0))
                if p_dir != 2 or edge < 8.0: continue
                
                tact = bar.get("tactical_probs", [0.33, 0.33, 0.33])
                meso = bar.get("meso_probs", [0.33, 0.33, 0.33])
                d_tact = tact[2] - tact[0]
                d_meso = meso[2] - meso[0]
                if d_tact < 0.05 or d_meso < 0.01: continue
                
                if len(rolling_15m) >= 30:
                    hi = max(rolling_15m); lo = min(rolling_15m); rng = hi - lo
                    if rng >= 20.0:
                        pos_in_rng = (mid - lo) / rng
                        if pos_in_rng >= 0.75: continue
                if last_exit_ts > 0 and (ts_ms - last_exit_ts) / 1000.0 < 200.0:
                    if mid > (last_exit_px - 10.0): continue
                pos = {"side": "BUY", "entry_px": ask, "bar": idx, "ts": ts_ms, "mfe": 0.0, "mae": 0.0}

        if pos is not None:
            epx = pos["entry_px"]
            gross = (bid - epx)/epx*10000.0
            net = gross - 4.0
            dur = (bars[-1].get("ts_ms", len(bars)*10000) - pos["ts"]) / 1000.0
            session_end_trades.append({
                "cycle": arch, "entry_bar": pos["bar"], "net": net, "dur": dur, "mfe": pos["mfe"], "mae": pos["mae"]
            })
            
    print("="*85)
    print(f"ANATOMY OF THE {len(session_end_trades)} TRAPPED TRADES IN LONG-ONLY MODE")
    print("="*85)
    print(f"{'Cycle':<18} | {'Entry Bar':<9} | {'Dur (s)':<10} | {'Net PnL':<10} | {'MAE':<8} | {'MFE':<8}")
    print("-" * 85)
    for t in session_end_trades:
        print(f"{t['cycle']:<18} | Bar {t['entry_bar']:<5} | {t['dur']:>8.1f}s | {t['net']:>+8.2f}b | {t['mae']:>+6.2f}b | {t['mfe']:>+6.2f}b")
    print(f"\nTotal Net PnL of these 12 trades: {sum(t['net'] for t in session_end_trades):+.2f} bps")

if __name__ == "__main__":
    inspect_long_only_session_end()
