import os
import glob
import json
import io
import sys

# Wrap stdout for safe utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

base_dir = "D:/PrivateProjects/BTC Futures/scratch"
archives = sorted([d for d in os.listdir(base_dir) if (d.startswith("archive_cycle") or d.startswith("archive_v3")) and os.path.isdir(os.path.join(base_dir, d))])

# Also include the latest cycle 64 live ledger if not in archive
cycle64_live = "D:/PrivateProjects/BTC Futures/scratch/live_hierarchical_session_1h_ledger.jsonl"

class UnifiedPeakReplayEngine:
    def __init__(self, shield_mfe=3.80, shield_net_lock=0.15, ratchet1_mfe=6.0, ratchet1_net=2.25, ratchet2_mfe=10.0, ratchet2_net=4.50, tp_target=35.0, wide_buffer=18.0):
        self.maker_fee = 2.0 # bps per leg (4.0 bps roundtrip)
        self.shield_mfe = shield_mfe
        self.shield_gross_floor = shield_net_lock + (2 * self.maker_fee) # e.g. 4.15 bps
        self.ratchet1_mfe = ratchet1_mfe
        self.ratchet1_gross_floor = ratchet1_net + (2 * self.maker_fee) # e.g. 6.25 bps
        self.ratchet2_mfe = ratchet2_mfe
        self.ratchet2_gross_floor = ratchet2_net + (2 * self.maker_fee) # e.g. 8.50 bps
        self.tp_target_gross = tp_target
        self.wide_buffer_gross = wide_buffer
        self.max_dur_sec = 600.0 # 10 mins

    def replay(self, ledger_path):
        if not os.path.exists(ledger_path):
            return None
            
        with open(ledger_path, "r", encoding="utf-8") as f:
            bars = [json.loads(line) for line in f if line.strip().startswith("{")]
            
        position = None
        closed_trades = []
        cumulative_pnl = 0.0
        
        last_exit_price = 0.0
        last_exit_side = ""
        last_exit_ts = 0
        
        for idx, bar in enumerate(bars):
            mid = bar.get("bbo_mid", bar.get("mid", 0.0))
            bid = bar.get("bbo_bid", mid - 0.05)
            ask = bar.get("bbo_ask", mid + 0.05)
            ts_ms = bar.get("ts_ms", idx * 10000)
            
            # 1. Manage Active Position
            if position is not None:
                pos_side = position["side"]
                entry_px = position["entry_price"]
                entry_ts = position["entry_ts"]
                duration = (ts_ms - entry_ts) / 1000.0
                
                # Gross return in bps
                if pos_side == "BUY":
                    cur_gross = (bid - entry_px) / entry_px * 10000.0
                else:
                    cur_gross = (entry_px - ask) / entry_px * 10000.0
                    
                position["mfe"] = max(position["mfe"], cur_gross)
                position["mae"] = min(position["mae"], cur_gross)
                mfe = position["mfe"]
                
                # Determine Active Ratchet Floor
                active_gross_floor = None
                ratchet_reason = None
                
                if mfe >= self.ratchet2_mfe:
                    active_gross_floor = self.ratchet2_gross_floor
                    ratchet_reason = "TIER2_RUNNER_RATCHET_TP"
                elif mfe >= self.ratchet1_mfe:
                    active_gross_floor = self.ratchet1_gross_floor
                    ratchet_reason = "TIER1_STEPPED_RATCHET_TP"
                elif mfe >= self.shield_mfe:
                    active_gross_floor = self.shield_gross_floor
                    ratchet_reason = "TIER0_BREAKEVEN_MAKER_SHIELD"
                    
                # A. Target TP reached
                if cur_gross >= self.tp_target_gross:
                    net_pnl = cur_gross - (2 * self.maker_fee)
                    closed_trades.append({
                        "side": pos_side, "entry": entry_px, "exit": mid,
                        "gross": cur_gross, "net": net_pnl, "dur": duration,
                        "mfe": mfe, "mae": position["mae"], "reason": "TARGET_MAKER_TP"
                    })
                    cumulative_pnl += net_pnl
                    last_exit_price = mid
                    last_exit_side = pos_side
                    last_exit_ts = ts_ms
                    position = None
                    continue
                    
                # B. Retrace into Ratchet Floor
                if active_gross_floor is not None and cur_gross <= active_gross_floor:
                    net_pnl = active_gross_floor - (2 * self.maker_fee)
                    closed_trades.append({
                        "side": pos_side, "entry": entry_px, "exit": mid,
                        "gross": active_gross_floor, "net": net_pnl, "dur": duration,
                        "mfe": mfe, "mae": position["mae"], "reason": ratchet_reason
                    })
                    cumulative_pnl += net_pnl
                    last_exit_price = mid
                    last_exit_side = pos_side
                    last_exit_ts = ts_ms
                    position = None
                    continue
                    
                # C. Model Reversal Harvest (if opposite model probability >= 0.48 and gross >= 4.0)
                tact = bar.get("tactical_probs", [0.33, 0.33, 0.33])
                opp_prob = tact[0] if pos_side == "BUY" else tact[2]
                if opp_prob >= 0.48 and cur_gross >= 4.0:
                    net_pnl = cur_gross - (2 * self.maker_fee)
                    closed_trades.append({
                        "side": pos_side, "entry": entry_px, "exit": mid,
                        "gross": cur_gross, "net": net_pnl, "dur": duration,
                        "mfe": mfe, "mae": position["mae"], "reason": "MODEL_OPPOSITE_HARVEST"
                    })
                    cumulative_pnl += net_pnl
                    last_exit_price = mid
                    last_exit_side = pos_side
                    last_exit_ts = ts_ms
                    position = None
                    continue
                    
                # D. Passive Maker Time Unwind
                if duration >= self.max_dur_sec and cur_gross >= 4.0:
                    net_pnl = cur_gross - (2 * self.maker_fee)
                    closed_trades.append({
                        "side": pos_side, "entry": entry_px, "exit": mid,
                        "gross": cur_gross, "net": net_pnl, "dur": duration,
                        "mfe": mfe, "mae": position["mae"], "reason": "PASSIVE_MAKER_TIME_UNWIND"
                    })
                    cumulative_pnl += net_pnl
                    last_exit_price = mid
                    last_exit_side = pos_side
                    last_exit_ts = ts_ms
                    position = None
                    continue
                    
                # E. Wide Passive Maker Buffer (only on severe breakdown)
                if cur_gross <= -self.wide_buffer_gross:
                    net_pnl = -self.wide_buffer_gross - (2 * self.maker_fee)
                    closed_trades.append({
                        "side": pos_side, "entry": entry_px, "exit": mid,
                        "gross": -self.wide_buffer_gross, "net": net_pnl, "dur": duration,
                        "mfe": mfe, "mae": position["mae"], "reason": "PASSIVE_MAKER_WIDE_BUFFER"
                    })
                    cumulative_pnl += net_pnl
                    last_exit_price = mid
                    last_exit_side = pos_side
                    last_exit_ts = ts_ms
                    position = None
                    continue
                    
            # 2. Evaluate IDLE Entries
            if position is None and idx >= 10:
                tact = bar.get("tactical_probs", [0.33, 0.33, 0.33])
                meso = bar.get("meso_probs", [0.33, 0.33, 0.33])
                micro = bar.get("micro_probs", [0.33, 0.33, 0.33])
                p_dir = bar.get("primary_direction", bar.get("direction", 1))
                edge = bar.get("expected_net_edge_bps", bar.get("edge", 0.0))
                
                d_tact = tact[2] - tact[0]
                d_micro = micro[2] - micro[0]
                d_meso = meso[2] - meso[0]
                
                # Check Tactical Concordance
                cand_dir = None
                tact_dir = 2 if d_tact >= 0.14 else (0 if d_tact <= -0.14 else 1)
                
                # Symmetrical confluence
                if tact_dir == 2 and d_micro > 0.01 and d_meso >= 0.00:
                    cand_dir = 2
                elif tact_dir == 0 and d_micro < -0.01 and d_meso <= 0.00:
                    cand_dir = 0
                elif p_dir == 2 and (edge >= 25.0 or (edge >= 4.0 and edge <= 20.0)) and d_meso >= 0.00:
                    cand_dir = 2
                elif p_dir == 0 and (edge >= 25.0 or (edge >= 4.0 and edge <= 20.0)) and d_meso <= 0.00:
                    cand_dir = 0
                    
                # Check Conformal Certainty Gate
                if cand_dir is not None:
                    tact_cert = tact[cand_dir]
                    meso_cert = meso[cand_dir]
                    if (tact_cert >= 0.40 or meso_cert >= 0.40):
                        # Same-Side Retrace cooldown barrier (Cycle 31 feature)
                        retrace_ok = True
                        if last_exit_price > 0.0:
                            elapsed_since_exit = (ts_ms - last_exit_ts) / 1000.0
                            if elapsed_since_exit < 180.0:
                                if cand_dir == 2 and last_exit_side == "BUY" and (mid > last_exit_price - 6.0):
                                    retrace_ok = False
                                elif cand_dir == 0 and last_exit_side == "SELL" and (mid < last_exit_price + 6.0):
                                    retrace_ok = False
                                    
                        if retrace_ok:
                            if cand_dir == 2:
                                position = {"side": "BUY", "entry_price": ask, "entry_ts": ts_ms, "mfe": 0.0, "mae": 0.0}
                            else:
                                position = {"side": "SELL", "entry_price": bid, "entry_ts": ts_ms, "mfe": 0.0, "mae": 0.0}
                                
        wins = len([t for t in closed_trades if t["net"] > 0])
        cnt = len(closed_trades)
        wr = (wins / cnt * 100.0) if cnt > 0 else 0.0
        avg_pnl = (cumulative_pnl / cnt) if cnt > 0 else 0.0
        
        gross_wins = sum([t["net"] for t in closed_trades if t["net"] > 0])
        gross_losses = abs(sum([t["net"] for t in closed_trades if t["net"] < 0]))
        pf = (gross_wins / gross_losses) if gross_losses > 0 else 999.00
        
        return {
            "trades": cnt,
            "wins": wins,
            "losses": cnt - wins,
            "win_rate": wr,
            "pnl": cumulative_pnl,
            "avg_pnl": avg_pnl,
            "pf": pf,
            "trades_list": closed_trades
        }

engine = UnifiedPeakReplayEngine()

print("="*110)
print("REPLAYING UNIFIED PEAK ARCHITECTURE ACROSS ALL HISTORICAL CYCLES")
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
    if not ledgers:
        continue
    ledger = ledgers[0]
    res = engine.replay(ledger)
    if res and res["trades"] > 0:
        all_results.append((arch, res))
        total_trades += res["trades"]
        total_wins += res["wins"]
        total_losses += res["losses"]
        total_net_pnl += res["pnl"]
        print(f"{arch:<24} | {res['trades']:<8} | {res['win_rate']:>5.1f}%     | {res['pnl']:>+12.2f} bps    | {res['avg_pnl']:>+8.2f} bps | {res['pf']:>6.2f}")

# Also run on Cycle 64 live if not yet in archive list
if os.path.exists(cycle64_live):
    res_c64 = engine.replay(cycle64_live)
    if res_c64 and res_c64["trades"] > 0:
        print(f"{'live_cycle64_session':<24} | {res_c64['trades']:<8} | {res_c64['win_rate']:>5.1f}%     | {res_c64['pnl']:>+12.2f} bps    | {res_c64['avg_pnl']:>+8.2f} bps | {res_c64['pf']:>6.2f}")

overall_wr = (total_wins / total_trades * 100.0) if total_trades > 0 else 0.0
overall_avg = (total_net_pnl / total_trades) if total_trades > 0 else 0.0

print("="*110)
print(f"{'ALL-TIME REPLAY TOTALS':<24} | {total_trades:<8} | {overall_wr:>5.1f}%     | {total_net_pnl:>+12.2f} bps    | {overall_avg:>+8.2f} bps |")
print("="*110)

# Summary of 100% win rate cycles
wr100_cycles = [a for a, r in all_results if r["win_rate"] == 100.0]
print(f"\nNumber of Cycles with 100.0% Win Rate under Unified Peak Engine: {len(wr100_cycles)} / {len(all_results)} ({len(wr100_cycles)/len(all_results)*100.0:.1f}%)")
