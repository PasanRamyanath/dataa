import os
import glob
import json
import collections
import io
import sys

# Ensure UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

base_dir = "D:/PrivateProjects/BTC Futures/scratch"
dirs = sorted([d for d in os.listdir(base_dir) if (d.startswith("archive_cycle") or d.startswith("archive_v3")) and os.path.isdir(os.path.join(base_dir, d))])

class FullUnifiedPeakReplayer:
    def __init__(self):
        self.maker_fee = 2.0
        self.shield_mfe = 3.80
        self.shield_floor = 4.15 # +0.15b net
        self.c48_mfe = 4.50
        self.c48_floor = 5.25   # +1.25b net (Cycle 48 Crown)
        self.ratchet1_mfe = 6.00
        self.ratchet1_floor = 6.25 # +2.25b net
        self.ratchet2_mfe = 10.00
        self.ratchet2_floor = 8.50 # +4.50b net
        self.ratchet3_mfe = 15.00
        self.ratchet3_floor = 12.50 # +8.50b net
        self.tp_target = 35.00  # +31.0b net
        self.catastrophic_breaker = 45.00
        self.max_dur = 900.0    # 15 minutes max holding

    def replay_ledger(self, ledger_file):
        with open(ledger_file, "r", encoding="utf-8") as f:
            bars = [json.loads(l) for l in f if l.strip().startswith("{")]
            
        rolling_15m = collections.deque(maxlen=90)
        pos = None
        closed = []
        cum_pnl = 0.0
        
        last_exit_ts = 0
        last_exit_side = ""
        last_exit_px = 0.0
        last_exit_was_win = True
        
        for idx, bar in enumerate(bars):
            mid = bar.get("bbo_mid", bar.get("mid", 0.0))
            bid = bar.get("bbo_bid", mid - 0.05)
            ask = bar.get("bbo_ask", mid + 0.05)
            ts_ms = bar.get("ts_ms", idx * 10000)
            rolling_15m.append(mid)
            
            # Position management
            if pos is not None:
                side = pos["side"]
                epx = pos["entry_px"]
                gross = (bid - epx)/epx*10000.0 if side == "BUY" else (epx - ask)/epx*10000.0
                pos["mfe"] = max(pos["mfe"], gross)
                pos["mae"] = min(pos["mae"], gross)
                mfe = pos["mfe"]
                dur = (ts_ms - pos["ts"]) / 1000.0
                
                # Dynamic Multi-Tier Ratchet Ladder
                floor = None
                reason = None
                if mfe >= self.ratchet3_mfe:
                    floor = self.ratchet3_floor
                    reason = "TIER3_RUNNER_EXPANSION"
                elif mfe >= self.ratchet2_mfe:
                    floor = self.ratchet2_floor
                    reason = "TIER2_STEPPED_RATCHET"
                elif mfe >= self.ratchet1_mfe:
                    floor = self.ratchet1_floor
                    reason = "TIER1_STEPPED_RATCHET"
                elif mfe >= self.c48_mfe:
                    floor = self.c48_floor
                    reason = "TIER0_5_C48_RATCHET"
                elif mfe >= self.shield_mfe:
                    floor = self.shield_floor
                    reason = "TIER0_BREAKEVEN_SHIELD"
                    
                # Full Target TP
                if gross >= self.tp_target:
                    net = gross - (2 * self.maker_fee)
                    closed.append({"side": side, "net": net, "dur": dur, "mfe": mfe, "mae": pos["mae"], "reason": "TARGET_MAKER_TP"})
                    cum_pnl += net
                    last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; last_exit_was_win = True; pos = None; continue
                # Ratchet Retrace Floor
                elif floor is not None and gross <= floor:
                    net = floor - (2 * self.maker_fee)
                    closed.append({"side": side, "net": net, "dur": dur, "mfe": mfe, "mae": pos["mae"], "reason": reason})
                    cum_pnl += net
                    last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; last_exit_was_win = True; pos = None; continue
                # Profitable Time Unwind
                elif dur >= self.max_dur and gross >= 4.0:
                    net = gross - (2 * self.maker_fee)
                    closed.append({"side": side, "net": net, "dur": dur, "mfe": mfe, "mae": pos["mae"], "reason": "TIME_UNWIND"})
                    cum_pnl += net
                    last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; last_exit_was_win = True; pos = None; continue
                # Structural Circuit Breaker (-45.0 bps Post-Only Maker limit)
                elif gross <= -self.catastrophic_breaker:
                    net = -self.catastrophic_breaker - (2 * self.maker_fee)
                    closed.append({"side": side, "net": net, "dur": dur, "mfe": mfe, "mae": pos["mae"], "reason": "CATASTROPHIC_BREAKER"})
                    cum_pnl += net
                    last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; last_exit_was_win = False; pos = None; continue
                    
            # IDLE Entries
            if pos is None and idx >= 15:
                tact = bar.get("tactical_probs", [0.33, 0.33, 0.33])
                meso = bar.get("meso_probs", [0.33, 0.33, 0.33])
                micro = bar.get("micro_probs", [0.33, 0.33, 0.33])
                p_dir = bar.get("primary_direction", bar.get("direction", 1))
                edge = bar.get("expected_net_edge_bps", bar.get("edge", 0.0))
                
                conf_tact = max(tact)
                conf_meso = max(meso)
                d_tact = tact[2] - tact[0]
                d_micro = micro[2] - micro[0]
                d_meso = meso[2] - meso[0]
                
                has_conviction = (conf_tact >= 0.42 and conf_meso >= 0.38) or (
                    abs(d_tact) >= 0.14 and abs(d_meso) >= 0.03
                )
                if not has_conviction:
                    continue
                    
                cand_dir = None
                tact_dir = 2 if d_tact >= 0.12 else (0 if d_tact <= -0.12 else 1)
                tact_confirmed = (tact_dir in (0, 2)) and (
                    (tact_dir == 2 and d_micro > 0.005 and d_meso >= 0.02) or
                    (tact_dir == 0 and d_micro < -0.005 and d_meso <= -0.02)
                )
                
                if tact_confirmed:
                    cand_dir = tact_dir
                elif p_dir in (0, 2) and edge >= 5.0:
                    if p_dir == 2 and d_meso >= 0.02: cand_dir = 2
                    elif p_dir == 0 and d_meso <= -0.02: cand_dir = 0
                    
                if cand_dir in (0, 2):
                    # Quantile Range Envelope Filter (0.12 <= Pos <= 0.88)
                    if len(rolling_15m) >= 30:
                        hi = max(rolling_15m)
                        lo = min(rolling_15m)
                        rng = hi - lo
                        if rng >= 20.0:
                            pos_in_rng = (mid - lo) / rng
                            if cand_dir == 2 and pos_in_rng >= 0.88: continue
                            if cand_dir == 0 and pos_in_rng <= 0.12: continue
                            
                    # Post-Exit Spacing and Same-Side Retrace Price Barrier
                    if last_exit_ts > 0:
                        elapsed = (ts_ms - last_exit_ts) / 1000.0
                        if not last_exit_was_win and elapsed < 120.0:
                            continue
                        elif last_exit_was_win and elapsed < 15.0:
                            continue
                        if elapsed < 200.0 and last_exit_px > 0:
                            if cand_dir == 2 and last_exit_side == "BUY" and mid > (last_exit_px - 10.0): continue
                            if cand_dir == 0 and last_exit_side == "SELL" and mid < (last_exit_px + 10.0): continue
                            
                    pos = {"side": "BUY" if cand_dir == 2 else "SELL", "entry_px": ask if cand_dir == 2 else bid, "ts": ts_ms, "mfe": 0.0, "mae": 0.0}

        wins = len([t for t in closed if t["net"] > 0])
        cnt = len(closed)
        wr = (wins / cnt * 100.0) if cnt > 0 else 0.0
        avg_pnl = (cum_pnl / cnt) if cnt > 0 else 0.0
        gross_w = sum([t["net"] for t in closed if t["net"] > 0])
        gross_l = abs(sum([t["net"] for t in closed if t["net"] < 0]))
        pf = (gross_w / max(1e-6, gross_l)) if gross_l > 0 else (999.0 if gross_w > 0 else 1.0)
        
        return {
            "bars": len(bars),
            "trades": cnt,
            "wins": wins,
            "losses": cnt - wins,
            "win_rate": wr,
            "net_pnl": cum_pnl,
            "avg_pnl": avg_pnl,
            "profit_factor": pf,
            "closed": closed
        }

replayer = FullUnifiedPeakReplayer()
full_audit = []

for d in dirs:
    arch_path = os.path.join(base_dir, d)
    ledgers = glob.glob(os.path.join(arch_path, "*ledger*.jsonl"))
    if not ledgers:
        continue
        
    # Check historical tournament winner in this archive
    hist_info = {"winner": "N/A", "hist_pnl": 0.0, "hist_wr": 0.0, "hist_trades": 0}
    res_path = os.path.join(arch_path, "champion_challenger_final_results.json")
    lb_path = os.path.join(arch_path, "champion_challenger_leaderboard.json")
    target_res = res_path if os.path.exists(res_path) else (lb_path if os.path.exists(lb_path) else None)
    
    if target_res:
        try:
            with open(target_res, "r", encoding="utf-8") as f:
                rdata = json.load(f)
            lb = rdata.get("final_leaderboard", rdata.get("leaderboard", []))
            if lb:
                valid_lb = [c for c in lb if c.get("trades", 0) > 0]
                if valid_lb:
                    top_c = sorted(valid_lb, key=lambda x: x.get("net_pnl_bps", -999), reverse=True)[0]
                    hist_info = {
                        "winner": top_c.get("name", "N/A"),
                        "hist_pnl": top_c.get("net_pnl_bps", 0.0),
                        "hist_wr": top_c.get("win_rate_pct", 0.0),
                        "hist_trades": top_c.get("trades", 0)
                    }
        except Exception:
            pass
            
    ledger_file = ledgers[0]
    replay_res = replayer.replay_ledger(ledger_file)
    
    full_audit.append({
        "archive": d,
        "ledger": os.path.basename(ledger_file),
        "bars": replay_res["bars"],
        "replay_trades": replay_res["trades"],
        "replay_wins": replay_res["wins"],
        "replay_losses": replay_res["losses"],
        "replay_wr": replay_res["win_rate"],
        "replay_pnl": replay_res["net_pnl"],
        "replay_avg": replay_res["avg_pnl"],
        "replay_pf": replay_res["profit_factor"],
        "closed_details": replay_res["closed"],
        "hist_trades": hist_info["hist_trades"],
        "hist_wr": hist_info["hist_wr"],
        "hist_pnl": hist_info["hist_pnl"]
    })

print(f"Successfully processed {len(full_audit)} cycle archive ledgers.")
print("\n" + "=" * 130)
print(f"{'Archive':<22} | {'Bars':<5} | {'Replay Tr':<10} | {'Win Rate':<10} | {'Replay Net PnL':<15} | {'Hist Live PnL':<15} | {'Delta vs Hist':<14} | {'PF':<6}")
print("=" * 130)

tot_replay_tr = sum(a["replay_trades"] for a in full_audit)
tot_replay_w = sum(a["replay_wins"] for a in full_audit)
tot_replay_net = sum(a["replay_pnl"] for a in full_audit)

for a in full_audit:
    delta = a["replay_pnl"] - a["hist_pnl"]
    status = f"{a['replay_wins']}/{a['replay_trades']} ({a['replay_wr']:>5.1f}%)" if a["replay_trades"] > 0 else "0 Tr (Idle)"
    print(f"{a['archive']:<22} | {a['bars']:<5} | {a['replay_trades']:<10} | {a['replay_wr']:>5.1f}%    | {a['replay_pnl']:>+10.2f} bps    | {a['hist_pnl']:>+10.2f} bps    | {delta:>+10.2f} bps   | {a['replay_pf']:>6.2f}")

print("=" * 130)
ov_wr = (tot_replay_w / tot_replay_tr * 100.0) if tot_replay_tr > 0 else 0.0
print(f"TOTALS ({len(full_audit)} Archives)    | {'ALL':<5} | {tot_replay_tr:<10} | {ov_wr:>5.1f}%    | {tot_replay_net:>+10.2f} bps    |")
print("=" * 130)

