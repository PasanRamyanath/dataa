import json, os, glob

with open('scratch/archive_cycle43/live_hierarchical_session_1h_ledger.jsonl', 'r') as f:
    bars = [json.loads(l) for l in f if l.strip().startswith('{')]

# Replay on cycle 43


# Let's see what trades were generated in cycle 43
cfg = {"name": "Taker -45.0 bps", "sl_bps": 45.0, "is_taker": True}
# Run replay on cycle 43
import collections
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
            closed.append({"entry_bar": pos["bar"], "exit_bar": idx, "side": side, "net": gross - 4.0, "reason": "TP"})
            last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue
        elif floor_gross is not None and gross <= floor_gross:
            closed.append({"entry_bar": pos["bar"], "exit_bar": idx, "side": side, "net": floor_gross - 4.0, "reason": tier_name})
            last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue
        elif dur >= 900.0 and gross >= 4.0:
            closed.append({"entry_bar": pos["bar"], "exit_bar": idx, "side": side, "net": gross - 4.0, "reason": "UNWIND"})
            last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue
        elif gross <= -45.0:
            closed.append({"entry_bar": pos["bar"], "exit_bar": idx, "side": side, "net": gross - 6.5, "reason": "STOP_LOSS"})
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
            if len(rolling_15m) >= 30:
                hi = max(rolling_15m); lo = min(rolling_15m); rng = hi - lo
                if rng >= 20.0:
                    pos_in_rng = (mid - lo) / rng
                    if cand_side == "BUY" and pos_in_rng >= 0.80: continue
                    if cand_side == "SELL" and pos_in_rng <= 0.20: continue
            if last_exit_ts > 0 and (ts_ms - last_exit_ts) / 1000.0 < 200.0:
                if cand_side == "BUY" and last_exit_side == "BUY" and mid > (last_exit_px - 10.0): continue
                if cand_side == "SELL" and last_exit_side == "SELL" and mid < (last_exit_px + 10.0): continue
            pos = {"side": cand_side, "entry_px": ask if cand_side == "BUY" else bid, "bar": idx, "ts": ts_ms, "mfe": 0.0, "mae": 0.0}

print("Closed trades in Cycle 43:")
for c in closed:
    print(" ", c)
