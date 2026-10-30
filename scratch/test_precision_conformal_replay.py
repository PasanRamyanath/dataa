import os
import glob
import json
import io
import sys

base_dir = "D:/PrivateProjects/BTC Futures/scratch"
archives = sorted([d for d in os.listdir(base_dir) if (d.startswith("archive_cycle") or d.startswith("archive_v3")) and os.path.isdir(os.path.join(base_dir, d))])

class PrecisionPeakEngine:
    def __init__(self, conformal_gate=0.45, shield_mfe=3.80, shield_net_lock=0.15, ratchet1_mfe=6.0, ratchet1_net=2.25, ratchet2_mfe=10.0, ratchet2_net=4.50, tp_target=35.0, wide_buffer=14.5):
        self.maker_fee = 2.0
        self.conformal_gate = conformal_gate
        self.shield_mfe = shield_mfe
        self.shield_gross_floor = shield_net_lock + (2 * self.maker_fee) # 4.15 bps
        self.ratchet1_mfe = ratchet1_mfe
        self.ratchet1_gross_floor = ratchet1_net + (2 * self.maker_fee) # 6.25 bps
        self.ratchet2_mfe = ratchet2_mfe
        self.ratchet2_gross_floor = ratchet2_net + (2 * self.maker_fee) # 8.50 bps
        self.tp_target_gross = tp_target
        self.wide_buffer_gross = wide_buffer
        self.max_dur_sec = 600.0

    def replay(self, ledger_path):
        if not os.path.exists(ledger_path): return None
        with open(ledger_path, "r", encoding="utf-8") as f:
            bars = [json.loads(l) for l in f if l.strip().startswith("{")]
            
        position = None
        closed = []
        cum_pnl = 0.0
        
        last_loss_ts = 0
        last_loss_side = ""
        last_loss_px = 0.0
        
        for idx, bar in enumerate(bars):
            mid = bar.get("bbo_mid", bar.get("mid", 0.0))
            bid = bar.get("bbo_bid", mid - 0.05)
            ask = bar.get("bbo_ask", mid + 0.05)
            ts_ms = bar.get("ts_ms", idx * 10000)
            
            # Position Management
            if position is not None:
                side = position["side"]
                epx = position["entry_price"]
                gross = (bid - epx)/epx*10000.0 if side == "BUY" else (epx - ask)/epx*10000.0
                position["mfe"] = max(position["mfe"], gross)
                position["mae"] = min(position["mae"], gross)
                mfe = position["mfe"]
                dur = (ts_ms - position["entry_ts"]) / 1000.0
                
                floor = None
                reason = None
                if mfe >= self.ratchet2_mfe:
                    floor = self.ratchet2_gross_floor
                    reason = "TIER2_RUNNER_RATCHET_TP"
                elif mfe >= self.ratchet1_mfe:
                    floor = self.ratchet1_gross_floor
                    reason = "TIER1_STEPPED_RATCHET_TP"
                elif mfe >= self.shield_mfe:
                    floor = self.shield_gross_floor
                    reason = "TIER0_BREAKEVEN_MAKER_SHIELD"
                    
                if gross >= self.tp_target_gross:
                    net = gross - (2 * self.maker_fee)
                    closed.append({"side": side, "net": net, "dur": dur, "mfe": mfe, "mae": position["mae"], "reason": "TARGET_MAKER_TP"})
                    cum_pnl += net
                    position = None
                    continue
                elif floor is not None and gross <= floor:
                    net = floor - (2 * self.maker_fee)
                    closed.append({"side": side, "net": net, "dur": dur, "mfe": mfe, "mae": position["mae"], "reason": reason})
                    cum_pnl += net
                    position = None
                    continue
                elif dur >= self.max_dur_sec and gross >= 4.0:
                    net = gross - (2 * self.maker_fee)
                    closed.append({"side": side, "net": net, "dur": dur, "mfe": mfe, "mae": position["mae"], "reason": "TIME_UNWIND"})
                    cum_pnl += net
                    position = None
                    continue
                elif gross <= -self.wide_buffer_gross:
                    net = -self.wide_buffer_gross - (2 * self.maker_fee)
                    closed.append({"side": side, "net": net, "dur": dur, "mfe": mfe, "mae": position["mae"], "reason": "PASSIVE_SL"})
                    cum_pnl += net
                    last_loss_ts = ts_ms
                    last_loss_side = side
                    last_loss_px = mid
                    position = None
                    continue

            # IDLE Entry Evaluation
            if position is None and idx >= 10:
                tact = bar.get("tactical_probs", [0.33, 0.33, 0.33])
                meso = bar.get("meso_probs", [0.33, 0.33, 0.33])
                p_dir = bar.get("primary_direction", bar.get("direction", 1))
                edge = bar.get("expected_net_edge_bps", bar.get("edge", 0.0))
                
                # Strict Cycle 35 Conformal Softmax Certainty
                # Must have softmax certainty >= conformal_gate (0.45) on at least one tier
                tact_max = max(tact[0], tact[2])
                meso_max = max(meso[0], meso[2])
                if tact_max < self.conformal_gate and meso_max < self.conformal_gate:
                    continue
                    
                cand_dir = None
                d_tact = tact[2] - tact[0]
                d_meso = meso[2] - meso[0]
                
                if d_tact >= 0.14 and d_meso >= 0.00 and tact[2] >= self.conformal_gate:
                    cand_dir = 2
                elif d_tact <= -0.14 and d_meso <= 0.00 and tact[0] >= self.conformal_gate:
                    cand_dir = 0
                elif p_dir == 2 and (edge >= 28.0 or (edge >= 5.0 and edge <= 20.0)) and d_meso >= 0.00 and meso[2] >= self.conformal_gate:
                    cand_dir = 2
                elif p_dir == 0 and (edge >= 28.0 or (edge >= 5.0 and edge <= 20.0)) and d_meso <= 0.00 and meso[0] >= self.conformal_gate:
                    cand_dir = 0
                    
                if cand_dir is not None:
                    # Post-loss spacing veto (Cycle 63 Rank 1 logic): 
                    # Do not re-enter the same direction for 300s after a loss unless price retraced past exit
                    if last_loss_ts > 0 and (ts_ms - last_loss_ts) / 1000.0 < 300.0:
                        if cand_dir == 2 and last_loss_side == "BUY" and mid < last_loss_px:
                            continue
                        if cand_dir == 0 and last_loss_side == "SELL" and mid > last_loss_px:
                            continue
                            
                    if cand_dir == 2:
                        position = {"side": "BUY", "entry_price": ask, "entry_ts": ts_ms, "mfe": 0.0, "mae": 0.0}
                    else:
                        position = {"side": "SELL", "entry_price": bid, "entry_ts": ts_ms, "mfe": 0.0, "mae": 0.0}
                        
        wins = len([t for t in closed if t["net"] > 0])
        cnt = len(closed)
        wr = (wins / cnt * 100.0) if cnt > 0 else 0.0
        avg_pnl = (cum_pnl / cnt) if cnt > 0 else 0.0
        
        gross_w = sum([t["net"] for t in closed if t["net"] > 0])
        gross_l = abs(sum([t["net"] for t in closed if t["net"] < 0]))
        pf = (gross_w / gross_l) if gross_l > 0 else 999.00
        
        return {"trades": cnt, "wins": wins, "losses": cnt - wins, "win_rate": wr, "wr": wr, "pnl": cum_pnl, "avg_pnl": avg_pnl, "pf": pf, "trades_list": closed}

engine = PrecisionPeakEngine()
print("="*110)
print("REPLAYING PRECISION CONFORMAL PEAK ENGINE ACROSS ALL HISTORICAL CYCLES")
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
print(f"\nTotal Active Cycles: {len(all_results)}")
print(f"Cycles with 100.0% Win Rate: {len(wr100_cycles)} / {len(all_results)} ({len(wr100_cycles)/len(all_results)*100.0:.1f}%)")
print(f"Cycles strictly Net Profitable: {len(profitable_cycles)} / {len(all_results)} ({len(profitable_cycles)/len(all_results)*100.0:.1f}%)")
