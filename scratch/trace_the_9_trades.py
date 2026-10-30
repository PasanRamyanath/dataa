import os
import glob
import json
import collections
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

base_dir = "D:/PrivateProjects/BTC Futures/scratch"

def trace_the_9_trades_in_config_c():
    target_cycles = [
        "archive_cycle17", "archive_cycle19", "archive_cycle38", "archive_cycle40", 
        "archive_cycle43", "archive_cycle52", "archive_cycle57", "archive_cycle58", "archive_cycle60"
    ]
    
    maker_fee = 2.0
    
    for arch in target_cycles:
        path = os.path.join(base_dir, arch)
        ledgers = glob.glob(os.path.join(path, "*ledger*.jsonl"))
        if not ledgers: continue
        with open(ledgers[0], "r", encoding="utf-8") as f:
            bars = [json.loads(l) for l in f if l.strip().startswith("{")]
            
        rolling_15m = collections.deque(maxlen=90)
        pos = None
        closed = []
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
                tier_name = None
                if mfe >= 15.0: floor_gross = 12.50; tier_name = "TIER3"
                elif mfe >= 10.0: floor_gross = 8.50; tier_name = "TIER2"
                elif mfe >= 6.0: floor_gross = 6.25; tier_name = "TIER1"
                elif mfe >= 4.5: floor_gross = 5.25; tier_name = "TIER0_5"
                elif mfe >= 3.8: floor_gross = 4.15; tier_name = "TIER0"
                
                if gross >= 35.0:
                    net = gross - 4.0; closed.append({"side": side, "entry_bar": pos["bar"], "exit_bar": idx, "net": net, "reason": "TP", "dur": dur, "mfe": mfe, "mae": pos["mae"]})
                    last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue
                elif floor_gross is not None and gross <= floor_gross:
                    net = floor_gross - 4.0; closed.append({"side": side, "entry_bar": pos["bar"], "exit_bar": idx, "net": net, "reason": tier_name, "dur": dur, "mfe": mfe, "mae": pos["mae"]})
                    last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue
                elif dur >= 900.0 and gross >= 4.0:
                    net = gross - 4.0; closed.append({"side": side, "entry_bar": pos["bar"], "exit_bar": idx, "net": net, "reason": "PROFIT_UNWIND", "dur": dur, "mfe": mfe, "mae": pos["mae"]})
                    last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue

            if pos is None and idx >= 15:
                p_dir = bar.get("primary_direction", bar.get("direction", 1))
                edge = bar.get("expected_net_edge_bps", bar.get("edge", 0.0))
                if p_dir == 1 or edge < 5.0: continue
                
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

        # Check what happened at the end of the session
        print(f"\n--- Cycle: {arch} ---")
        if closed:
            for c in closed:
                print(f"  Closed Trade: Side={c['side']} Entry={c['entry_bar']} Exit={c['exit_bar']} Net={c['net']:+.2f}b Dur={c['dur']:.1f}s MFE={c['mfe']:.2f}b MAE={c['mae']:.2f}b Reason={c['reason']}")
        if pos is not None:
            final_gross = (bid - pos["entry_px"])/pos["entry_px"]*10000.0 if pos["side"] == "BUY" else (pos["entry_px"] - ask)/pos["entry_px"]*10000.0
            print(f"  *STILL OPEN AT SESSION END*: Side={pos['side']} Entry={pos['bar']} FinalGross={final_gross:+.2f}b MFE={pos['mfe']:.2f}b MAE={pos['mae']:.2f}b")

if __name__ == "__main__":
    trace_the_9_trades_in_config_c()
