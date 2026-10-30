import os
import glob
import json
import collections
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

base_dir = "D:/PrivateProjects/BTC Futures/scratch"
archives = sorted([d for d in os.listdir(base_dir) if (d.startswith("archive_cycle") or d.startswith("archive_v3")) and os.path.isdir(os.path.join(base_dir, d))])

def test_long_only_closed_trades():
    maker_fee = 2.0
    taker_fee = 4.5
    
    ladder = [
        (35.00, 35.00, "TARGET_MAKER_TP"),
        (20.00, 16.00, "TIER3_RUNNER_EXPANSION"),
        (14.00, 11.00, "TIER2_RUNNER_RATCHET"),
        (9.00,  7.00,  "TIER1_STEPPED_RATCHET"),
        (6.50,  5.25,  "TIER0_5_CYCLE48_CROWN"),
        (5.00,  4.15,  "TIER0_BREAKEVEN_SHIELD")
    ]
    
    closed_trades = []
    active_cycles = 0
    wr100_cycles = 0
    profitable_cycles = 0
    reasons = collections.Counter()
    
    for arch in archives:
        ledgers = glob.glob(os.path.join(base_dir, arch, "*ledger*.jsonl"))
        if not ledgers: continue
        with open(ledgers[0], "r", encoding="utf-8") as f:
            bars = [json.loads(l) for l in f if l.strip().startswith("{")]
            
        rolling_15m = collections.deque(maxlen=90)
        pos = None
        session_closed = []
        cum_pnl = 0.0
        last_exit_ts = 0; last_exit_px = 0.0
        
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
                
                # 1. Target TP (+35.0b)
                if gross >= 35.0:
                    net = gross - 4.0
                    rec = {"cycle": arch, "net": net, "dur": dur, "mfe": mfe, "mae": pos["mae"], "reason": "TARGET_MAKER_TP"}
                    session_closed.append(rec); closed_trades.append(rec); cum_pnl += net; reasons["TARGET_MAKER_TP"] += 1
                    last_exit_ts = ts_ms; last_exit_px = mid; pos = None; continue
                    
                # 2. Retrace Ratchets
                active_floor = None
                tier_name = None
                for hurdle, floor, name in ladder:
                    if name == "TARGET_MAKER_TP": continue
                    if mfe >= hurdle:
                        active_floor = floor
                        tier_name = name
                        break
                        
                if active_floor is not None and gross <= active_floor:
                    net = active_floor - 4.0
                    rec = {"cycle": arch, "net": net, "dur": dur, "mfe": mfe, "mae": pos["mae"], "reason": tier_name}
                    session_closed.append(rec); closed_trades.append(rec); cum_pnl += net; reasons[tier_name] += 1
                    last_exit_ts = ts_ms; last_exit_px = mid; pos = None; continue
                    
                # 3. Profitable Time Unwind (>= 900s & gross >= 4.0b)
                elif dur >= 900.0 and gross >= 4.0:
                    net = gross - 4.0
                    rec = {"cycle": arch, "net": net, "dur": dur, "mfe": mfe, "mae": pos["mae"], "reason": "PROFITABLE_TIME_UNWIND"}
                    session_closed.append(rec); closed_trades.append(rec); cum_pnl += net; reasons["PROFITABLE_TIME_UNWIND"] += 1
                    last_exit_ts = ts_ms; last_exit_px = mid; pos = None; continue
                    
                # 4. Emergency Taker Breaker (-45.0b)
                elif gross <= -45.0:
                    actual_loss = min(-45.0, gross)
                    net = actual_loss - (maker_fee + taker_fee)
                    rec = {"cycle": arch, "net": net, "dur": dur, "mfe": mfe, "mae": pos["mae"], "reason": "HARD_STOP_LOSS_TAKER"}
                    session_closed.append(rec); closed_trades.append(rec); cum_pnl += net; reasons["HARD_STOP_LOSS_TAKER"] += 1
                    last_exit_ts = ts_ms; last_exit_px = mid; pos = None; continue

            # Entries (Bar <= 270, STRICTLY LONG-ONLY)
            if pos is None and idx >= 15 and idx <= 270:
                p_dir = bar.get("primary_direction", bar.get("direction", 1))
                edge = bar.get("expected_net_edge_bps", bar.get("edge", 0.0))
                if p_dir != 2 or edge < 8.0: continue
                
                tact = bar.get("tactical_probs", [0.33, 0.33, 0.33])
                meso = bar.get("meso_probs", [0.33, 0.33, 0.33])
                micro = bar.get("micro_probs", [0.33, 0.33, 0.33])
                d_tact = tact[2] - tact[0]
                d_meso = meso[2] - meso[0]
                d_micro = micro[2] - micro[0]
                
                if d_tact < 0.05 or d_meso < 0.01 or d_micro <= 0.0: continue
                
                if len(rolling_15m) >= 30:
                    hi = max(rolling_15m); lo = min(rolling_15m); rng = hi - lo
                    if rng >= 20.0:
                        pos_in_rng = (mid - lo) / rng
                        if pos_in_rng >= 0.75: continue
                if last_exit_ts > 0 and (ts_ms - last_exit_ts) / 1000.0 < 200.0:
                    if mid > (last_exit_px - 8.0): continue
                pos = {"entry_px": ask, "bar": idx, "ts": ts_ms, "mfe": 0.0, "mae": 0.0}

        if session_closed:
            active_cycles += 1
            w = sum(1 for c in session_closed if c["net"] > 0)
            if w == len(session_closed): wr100_cycles += 1
            if cum_pnl > 0: profitable_cycles += 1

    tot_w = sum(1 for t in closed_trades if t["net"] > 0)
    tot_pnl = sum(t["net"] for t in closed_trades)
    wr = (tot_w / len(closed_trades) * 100.0) if closed_trades else 0.0
    avg = (tot_pnl / len(closed_trades)) if closed_trades else 0.0
    
    print("="*100)
    print("EMPIRICAL PURE LONG-ONLY PERFORMANCE (ALL 64 HISTORICAL CYCLES)")
    print("="*100)
    print(f"Total Closed Trades: {len(closed_trades)} across {active_cycles} active sessions")
    print(f"Wins: {tot_w} | Losses: {len(closed_trades) - tot_w}")
    print(f"Win Rate: {wr:.2f}%")
    print(f"Total Net Realized PnL: {tot_pnl:+.2f} bps NET REALIZED PROFIT")
    print(f"Average Profit per Trade: {avg:+.2f} bps/trade")
    print(f"100% Win Rate Sessions: {wr100_cycles} / {active_cycles} ({wr100_cycles/active_cycles*100:.1f}%)")
    print(f"Profitable Sessions: {profitable_cycles} / {active_cycles} ({profitable_cycles/active_cycles*100:.1f}%)")
    print("\nExit Mechanism Distribution:")
    for r, cnt in reasons.items():
        sub_pnl = sum(t["net"] for t in closed_trades if t["reason"] == r)
        print(f"  {r:<30}: {cnt:>3} trades | Net PnL: {sub_pnl:>+8.2f} bps | Avg: {sub_pnl/cnt:>+5.2f} bps")
    print("="*100)

if __name__ == "__main__":
    test_long_only_closed_trades()
