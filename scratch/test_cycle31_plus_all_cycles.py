import os
import glob
import json
import collections

base_dir = "D:/PrivateProjects/BTC Futures/scratch"
archives = sorted([d for d in os.listdir(base_dir) if (d.startswith("archive_cycle") or d.startswith("archive_v3")) and os.path.isdir(os.path.join(base_dir, d))])

class Cycle31PlusEngine:
    def __init__(self, shield_mfe=3.80, shield_floor=4.15, ratchet1_mfe=6.00, ratchet1_floor=6.25, ratchet2_mfe=10.00, ratchet2_floor=8.50, tp_target=35.0, wide_buffer=18.0):
        self.shield_mfe = shield_mfe
        self.shield_floor = shield_floor
        self.ratchet1_mfe = ratchet1_mfe
        self.ratchet1_floor = ratchet1_floor
        self.ratchet2_mfe = ratchet2_mfe
        self.ratchet2_floor = ratchet2_floor
        self.tp_target = tp_target
        self.wide_buffer = wide_buffer

    def replay(self, ledger_path):
        if not os.path.exists(ledger_path): return None
        with open(ledger_path, "r", encoding="utf-8") as f:
            bars = [json.loads(l) for l in f if l.strip().startswith("{")]
            
        position = None
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
            
            # Position management
            if position is not None:
                side = position["side"]
                epx = position["entry_px"]
                gross = (bid - epx)/epx*10000.0 if side == "BUY" else (epx - ask)/epx*10000.0
                position["mfe"] = max(position["mfe"], gross)
                position["mae"] = min(position["mae"], gross)
                mfe = position["mfe"]
                dur = (ts_ms - position["ts"]) / 1000.0
                
                floor = None
                reason = None
                if mfe >= self.ratchet2_mfe:
                    floor = self.ratchet2_floor
                    reason = "TIER2_RUNNER_RATCHET_TP"
                elif mfe >= self.ratchet1_mfe:
                    floor = self.ratchet1_floor
                    reason = "TIER1_STEPPED_RATCHET_TP"
                elif mfe >= self.shield_mfe:
                    floor = self.shield_floor
                    reason = "TIER0_BREAKEVEN_MAKER_SHIELD"
                    
                if gross >= self.tp_target:
                    net = gross - 4.0
                    closed.append({"side": side, "net": net, "dur": dur, "mfe": mfe, "mae": position["mae"], "reason": "TARGET_MAKER_TP"})
                    cum_pnl += net
                    last_exit_ts = ts_ms
                    last_exit_side = side
                    last_exit_px = mid
                    position = None
                    continue
                elif floor is not None and gross <= floor:
                    net = floor - 4.0
                    closed.append({"side": side, "net": net, "dur": dur, "mfe": mfe, "mae": position["mae"], "reason": reason})
                    cum_pnl += net
                    last_exit_ts = ts_ms
                    last_exit_side = side
                    last_exit_px = mid
                    position = None
                    continue
                elif dur >= 450.0 and gross >= 4.0:
                    net = gross - 4.0
                    closed.append({"side": side, "net": net, "dur": dur, "mfe": mfe, "mae": position["mae"], "reason": "TIME_UNWIND"})
                    cum_pnl += net
                    last_exit_ts = ts_ms
                    last_exit_side = side
                    last_exit_px = mid
                    position = None
                    continue
                elif gross <= -self.wide_buffer:
                    net = -self.wide_buffer - 4.0
                    closed.append({"side": side, "net": net, "dur": dur, "mfe": mfe, "mae": position["mae"], "reason": "PASSIVE_SL"})
                    cum_pnl += net
                    last_exit_ts = ts_ms
                    last_exit_side = side
                    last_exit_px = mid
                    position = None
                    continue

            # IDLE Entries (Cycle 31 Pure Logic)
            if position is None and idx >= 10:
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
                
                if tact_confirmed:
                    cand_dir = tact_dir
                elif p_dir in (0, 2) and (edge >= 28.0 or (edge >= 4.0 and edge <= 20.0)):
                    if p_dir == 2 and d_meso >= 0.00:
                        cand_dir = 2
                    elif p_dir == 0 and d_meso <= 0.00:
                        cand_dir = 0
                        
                if cand_dir in (0, 2):
                    # Same-side retrace barrier: $15.00 pullback before re-entering same side within 300s (Cycle 31)
                    if last_exit_ts > 0:
                        elapsed = (ts_ms - last_exit_ts) / 1000.0
                        if elapsed < 300.0 and last_exit_px > 0:
                            if cand_dir == 2 and last_exit_side == "BUY" and mid > (last_exit_px - 15.0):
                                continue
                            if cand_dir == 0 and last_exit_side == "SELL" and mid < (last_exit_px + 15.0):
                                continue
                                
                    if cand_dir == 2:
                        position = {"side": "BUY", "entry_px": ask, "ts": ts_ms, "mfe": 0.0, "mae": 0.0}
                    else:
                        position = {"side": "SELL", "entry_px": bid, "ts": ts_ms, "mfe": 0.0, "mae": 0.0}
                        
        wins = len([t for t in closed if t["net"] > 0])
        cnt = len(closed)
        wr = (wins / cnt * 100.0) if cnt > 0 else 0.0
        avg_pnl = (cum_pnl / cnt) if cnt > 0 else 0.0
        
        gross_w = sum([t["net"] for t in closed if t["net"] > 0])
        gross_l = abs(sum([t["net"] for t in closed if t["net"] < 0]))
        pf = (gross_w / gross_l) if gross_l > 0 else 999.00
        
        return {"trades": cnt, "wins": wins, "losses": cnt - wins, "win_rate": wr, "wr": wr, "pnl": cum_pnl, "avg_pnl": avg_pnl, "pf": pf, "trades_list": closed}

engine = Cycle31PlusEngine()
print("="*110)
print("REPLAYING EXACT CYCLE 31 ENGINE (+ Stepped Ratchets) ACROSS 63 CYCLES")
print("="*110)
print(f"{'Cycle Archive':<24} | {'Trades':<8} | {'Win Rate':<10} | {'Net Realized PnL':<18} | {'Avg/Trade':<12} | {'PF':<8}")
print("-" * 110)

all_results = []
total_trades = 0
total_wins = 0
total_losses = 0
total_net_pnl = 0.0

for arch in archives:
    p = os.path.join(base_dir, arch)
    ledgers = glob.glob(os.path.join(p, "*ledger*.jsonl"))
    if not ledgers: continue
    res = engine.replay(ledgers[0])
    if res and res["trades"] > 0:
        all_results.append((arch, res))
        total_trades += res["trades"]
        total_wins += res["wins"]
        total_losses += res["losses"]
        total_net_pnl += res["pnl"]
        print(f"{arch:<24} | {res['trades']:<8} | {res['win_rate']:>5.1f}%     | {res['pnl']:>+12.2f} bps    | {res['avg_pnl']:>+8.2f} bps | {res['pf']:>6.2f}")

overall_wr = (total_wins / total_trades * 100.0) if total_trades > 0 else 0.0
overall_avg = (total_net_pnl / total_trades) if total_trades > 0 else 0.0

print("="*110)
print(f"{'ALL-TIME REPLAY TOTALS':<24} | {total_trades:<8} | {overall_wr:>5.1f}%     | {total_net_pnl:>+12.2f} bps    | {overall_avg:>+8.2f} bps |")
print("="*110)

wr100_cycles = [a for a, r in all_results if r["win_rate"] == 100.0]
profitable_cycles = [a for a, r in all_results if r["pnl"] > 0.0]
print(f"\nTotal Active Cycles: {len(all_results)} / 63")
print(f"Cycles with 100.0% Win Rate: {len(wr100_cycles)} / {len(all_results)} ({len(wr100_cycles)/len(all_results)*100.0:.1f}%)")
print(f"Cycles strictly Net Profitable: {len(profitable_cycles)} / {len(all_results)} ({len(profitable_cycles)/len(all_results)*100.0:.1f}%)")
