import os
import glob
import json
import collections
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

base_dir = "D:/PrivateProjects/BTC Futures/scratch"

def inspect_the_7_drop_trades():
    target_cycles = ["archive_cycle38", "archive_cycle40", "archive_cycle43", "archive_cycle50", "archive_cycle52", "archive_cycle57", "archive_cycle58"]
    
    for tc in target_cycles:
        path = os.path.join(base_dir, tc)
        ledgers = glob.glob(os.path.join(path, "*ledger*.jsonl"))
        if not ledgers: continue
        with open(ledgers[0], "r", encoding="utf-8") as f:
            bars = [json.loads(l) for l in f if l.strip().startswith("{")]
            
        print(f"\n=======================================================")
        print(f"CYCLE: {tc} (Total Bars: {len(bars)})")
        print(f"=======================================================")
        
        rolling_15m = collections.deque(maxlen=90)
        pos = None
        
        for idx, bar in enumerate(bars):
            mid = bar.get("bbo_mid", bar.get("mid", 0.0))
            bid = bar.get("bbo_bid", mid - 0.05)
            ask = bar.get("bbo_ask", mid + 0.05)
            ts_ms = bar.get("ts_ms", idx * 10000)
            rolling_15m.append(mid)
            
            if pos is not None:
                side = pos["side"]
                epx = pos["entry_px"]
                gross = (bid - epx)/epx*10000.0 if side == "BUY" else (epx - ask)/epx*10000.0
                pos["mfe"] = max(pos["mfe"], gross)
                pos["mae"] = min(pos["mae"], gross)
                pos["trajectory"].append((idx, round(gross, 2)))
                
                # Check when it hits -45b
                if gross <= -45.0 and not pos.get("hit_45"):
                    pos["hit_45"] = True
                    pos["hit_45_bar"] = idx
                    pos["hit_45_gross"] = gross
                    
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
                    if len(rolling_15m) >= 30:
                        hi = max(rolling_15m); lo = min(rolling_15m); rng = hi - lo
                        if rng >= 20.0:
                            pos_in_rng = (mid - lo) / rng
                            if cand_side == "BUY" and pos_in_rng >= 0.80: continue
                            if cand_side == "SELL" and pos_in_rng <= 0.20: continue
                    pos = {"side": cand_side, "entry_px": ask if cand_side == "BUY" else bid, "entry_bar": idx, "mfe": 0.0, "mae": 0.0, "trajectory": [], "hit_45": False}
                    
            # If position finished session
            if idx == len(bars) - 1 and pos is not None:
                final_gross = pos["trajectory"][-1][1]
                min_gross = min(t[1] for t in pos["trajectory"])
                max_gross = max(t[1] for t in pos["trajectory"])
                print(f"Trade: Side={pos['side']}, EntryBar={pos['entry_bar']}, Duration={len(pos['trajectory'])*10}s")
                print(f"  MFE={max_gross:.2f}b, MAE={min_gross:.2f}b, FinalGross={final_gross:.2f}b")
                print(f"  Did it hit -45b? {pos['hit_45']} (At bar {pos.get('hit_45_bar')}, gross={pos.get('hit_45_gross', 0):.2f}b)")
                # Did it eventually recover after hitting -45b?
                if pos['hit_45']:
                    after_45 = [t[1] for t in pos["trajectory"] if t[0] >= pos["hit_45_bar"]]
                    print(f"  Trajectory after hitting -45b: Max={max(after_45):.2f}b, Final={after_45[-1]:.2f}b")
                # Print trajectory sample
                traj_sample = [f"Bar{t[0]}:{t[1]}b" for i, t in enumerate(pos["trajectory"]) if i % 15 == 0 or t[1] == min_gross or t[1] == max_gross]
                print(f"  Trajectory Highlights: {' -> '.join(traj_sample[:8])}")

if __name__ == "__main__":
    inspect_the_7_drop_trades()
