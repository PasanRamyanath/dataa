import os
import glob
import json
import collections
import io
import sys

# Ensure UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

base_dir = "D:/PrivateProjects/BTC Futures/scratch"
archives = sorted([d for d in os.listdir(base_dir) if (d.startswith("archive_cycle") or d.startswith("archive_v3")) and os.path.isdir(os.path.join(base_dir, d))])

class UnifiedPeakBenchmarkEngine:
    def __init__(
        self,
        fix1_taker_stop: bool = False,
        fix2_dead_trade_inval_sec: float = None,
        fix2_mfe_hurdle: float = 3.5,
        fix3_max_entry_bar: int = 360,
        fix4_no_flat: bool = False,
        fix4_min_edge: float = 0.0,
        fix4_q_lo: float = 0.20,
        fix4_q_hi: float = 0.80,
        sl_cutoff_bps: float = 45.0
    ):
        self.maker_fee = 2.0
        self.taker_fee = 4.5
        
        # Ratchet Ladder
        self.shield_mfe = 3.80
        self.shield_floor = 4.15 # +0.15b net
        self.c48_mfe = 4.50
        self.c48_floor = 5.25   # +1.25b net
        self.ratchet1_mfe = 6.00
        self.ratchet1_floor = 6.25 # +2.25b net
        self.ratchet2_mfe = 10.00
        self.ratchet2_floor = 8.50 # +4.50b net
        self.ratchet3_mfe = 15.00
        self.ratchet3_floor = 12.50 # +8.50b net
        self.tp_target = 35.00  # +31.0b net
        
        # Fix 1: Emergency stop execution
        self.fix1_taker_stop = fix1_taker_stop
        self.sl_cutoff_bps = sl_cutoff_bps
        
        # Fix 2: Dead-trade invalidation
        self.fix2_dead_trade_inval_sec = fix2_dead_trade_inval_sec
        self.fix2_mfe_hurdle = fix2_mfe_hurdle
        
        # Fix 3: Runway / Session-end freeze
        self.fix3_max_entry_bar = fix3_max_entry_bar
        
        # Fix 4: Directional / Edge discipline
        self.fix4_no_flat = fix4_no_flat
        self.fix4_min_edge = fix4_min_edge
        self.fix4_q_lo = fix4_q_lo
        self.fix4_q_hi = fix4_q_hi
        
    def evaluate_all_cycles(self):
        total_trades = []
        active_sessions = 0
        wr100_sessions = 0
        profitable_sessions = 0
        reasons = collections.Counter()
        
        cycle_summaries = {}
        
        for arch in archives:
            ledgers = glob.glob(os.path.join(base_dir, arch, "*ledger*.jsonl"))
            if not ledgers: continue
            with open(ledgers[0], "r", encoding="utf-8") as f:
                bars = [json.loads(l) for l in f if l.strip().startswith("{")]
                
            rolling_15m = collections.deque(maxlen=90)
            pos = None
            closed = []
            cum_pnl = 0.0
            last_exit_ts = 0; last_exit_side = ""; last_exit_px = 0.0
            
            for idx, bar in enumerate(bars):
                mid = bar.get("bbo_mid", bar.get("mid", 0.0))
                bid = bar.get("bbo_bid", mid - 0.05)
                ask = bar.get("bbo_ask", mid + 0.05)
                ts_ms = bar.get("ts_ms", idx * 10000)
                rolling_15m.append(mid)
                
                # Position management
                if pos is not None:
                    side = pos["side"]; epx = pos["entry_px"]
                    # Gross calculation at BBO:
                    gross = (bid - epx)/epx*10000.0 if side == "BUY" else (epx - ask)/epx*10000.0
                    pos["mfe"] = max(pos["mfe"], gross)
                    pos["mae"] = min(pos["mae"], gross)
                    mfe = pos["mfe"]; dur = (ts_ms - pos["ts"]) / 1000.0
                    
                    floor_gross = None
                    tier_name = None
                    if mfe >= self.ratchet3_mfe: floor_gross = self.ratchet3_floor; tier_name = "TIER3_RUNNER"
                    elif mfe >= self.ratchet2_mfe: floor_gross = self.ratchet2_floor; tier_name = "TIER2_RUNNER"
                    elif mfe >= self.ratchet1_mfe: floor_gross = self.ratchet1_floor; tier_name = "TIER1_STEP"
                    elif mfe >= self.c48_mfe: floor_gross = self.c48_floor; tier_name = "TIER0_5_C48"
                    elif mfe >= self.shield_mfe: floor_gross = self.shield_floor; tier_name = "TIER0_SHIELD"
                    
                    # 1. Target Alpha TP (+35 bps gross Maker limit)
                    if gross >= self.tp_target:
                        net = gross - (2 * self.maker_fee)
                        rec = {"cycle": arch, "side": side, "net": net, "dur": dur, "mfe": mfe, "mae": pos["mae"], "reason": "TARGET_TP", "entry_bar": pos["bar"], "exit_bar": idx}
                        closed.append(rec); total_trades.append(rec); cum_pnl += net; reasons["TARGET_TP"] += 1
                        last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue
                        
                    # 2. Stepped Ratchet Retrace Floor (Maker limit)
                    elif floor_gross is not None and gross <= floor_gross:
                        net = floor_gross - (2 * self.maker_fee)
                        rec = {"cycle": arch, "side": side, "net": net, "dur": dur, "mfe": mfe, "mae": pos["mae"], "reason": tier_name, "entry_bar": pos["bar"], "exit_bar": idx}
                        closed.append(rec); total_trades.append(rec); cum_pnl += net; reasons[tier_name] += 1
                        last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue
                        
                    # 3. Profitable Time Unwind (>= 900s and gross >= 4.0b)
                    elif dur >= 900.0 and gross >= 4.0:
                        net = gross - (2 * self.maker_fee)
                        rec = {"cycle": arch, "side": side, "net": net, "dur": dur, "mfe": mfe, "mae": pos["mae"], "reason": "PROFIT_UNWIND", "entry_bar": pos["bar"], "exit_bar": idx}
                        closed.append(rec); total_trades.append(rec); cum_pnl += net; reasons["PROFIT_UNWIND"] += 1
                        last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue
                        
                    # 4. FIX 2: Dead-Trade Early Invalidation (MFE < hurdle after X seconds)
                    elif self.fix2_dead_trade_inval_sec is not None and dur >= self.fix2_dead_trade_inval_sec and mfe < self.fix2_mfe_hurdle:
                        # Exiting dead trade via Maker BBO limit
                        net = gross - (2 * self.maker_fee)
                        rec = {"cycle": arch, "side": side, "net": net, "dur": dur, "mfe": mfe, "mae": pos["mae"], "reason": "DEAD_TRADE_INVAL", "entry_bar": pos["bar"], "exit_bar": idx}
                        closed.append(rec); total_trades.append(rec); cum_pnl += net; reasons["DEAD_TRADE_INVAL"] += 1
                        last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue

                    # 5. FIX 1: Emergency Stop Loss Breaker
                    elif gross <= -self.sl_cutoff_bps:
                        actual_loss = min(-self.sl_cutoff_bps, gross)
                        if self.fix1_taker_stop:
                            # Taker Stop Market crossing spread: maker entry + taker exit fee
                            net = actual_loss - (self.maker_fee + self.taker_fee)
                        else:
                            # Passive Maker limit (baseline)
                            net = actual_loss - (2 * self.maker_fee)
                        rec = {"cycle": arch, "side": side, "net": net, "dur": dur, "mfe": mfe, "mae": pos["mae"], "reason": "STOP_LOSS", "entry_bar": pos["bar"], "exit_bar": idx}
                        closed.append(rec); total_trades.append(rec); cum_pnl += net; reasons["STOP_LOSS"] += 1
                        last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; pos = None; continue

                # Entries (IDLE state)
                # FIX 3: Respect max entry bar (runway freeze)
                if pos is None and idx >= 15 and idx <= self.fix3_max_entry_bar:
                    p_dir = bar.get("primary_direction", bar.get("direction", 1))
                    edge = bar.get("expected_net_edge_bps", bar.get("edge", 0.0))
                    
                    # FIX 4: Directional / Edge filters
                    if self.fix4_no_flat and p_dir == 1:
                        continue
                    if edge < self.fix4_min_edge:
                        continue
                        
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
                    
                    cand_side = None
                    if is_long: cand_side = "BUY"
                    elif is_short: cand_side = "SELL"
                    
                    if cand_side is not None:
                        # Quantile Envelope Guard
                        if len(rolling_15m) >= 30:
                            hi = max(rolling_15m); lo = min(rolling_15m); rng = hi - lo
                            if rng >= 20.0:
                                pos_in_rng = (mid - lo) / rng
                                if cand_side == "BUY" and pos_in_rng >= self.fix4_q_hi: continue
                                if cand_side == "SELL" and pos_in_rng <= self.fix4_q_lo: continue
                                
                        # Same-side retrace barrier
                        if last_exit_ts > 0 and (ts_ms - last_exit_ts) / 1000.0 < 200.0:
                            if cand_side == "BUY" and last_exit_side == "BUY" and mid > (last_exit_px - 10.0): continue
                            if cand_side == "SELL" and last_exit_side == "SELL" and mid < (last_exit_px + 10.0): continue
                            
                        entry_px = ask if cand_side == "BUY" else bid
                        pos = {"side": cand_side, "entry_px": entry_px, "bar": idx, "ts": ts_ms, "mfe": 0.0, "mae": 0.0}

            # Session-end mark to market (close any remaining open trade)
            if pos is not None:
                side = pos["side"]; epx = pos["entry_px"]
                gross = (bid - epx)/epx*10000.0 if side == "BUY" else (epx - ask)/epx*10000.0
                net = gross - (2 * self.maker_fee)
                dur = (bars[-1].get("ts_ms", len(bars)*10000) - pos["ts"]) / 1000.0
                rec = {"cycle": arch, "side": side, "net": net, "dur": dur, "mfe": pos["mfe"], "mae": pos["mae"], "reason": "SESSION_END_MARK", "entry_bar": pos["bar"], "exit_bar": len(bars)-1}
                closed.append(rec); total_trades.append(rec); cum_pnl += net; reasons["SESSION_END_MARK"] += 1

            if closed:
                active_sessions += 1
                w = sum(1 for c in closed if c["net"] > 0)
                if w == len(closed): wr100_sessions += 1
                if cum_pnl > 0: profitable_sessions += 1
                cycle_summaries[arch] = {"trades": len(closed), "wins": w, "net": cum_pnl}
                
        tot_w = sum(1 for t in total_trades if t["net"] > 0)
        tot_pnl = sum(t["net"] for t in total_trades)
        wr = (tot_w / len(total_trades) * 100.0) if total_trades else 0.0
        avg = (tot_pnl / len(total_trades)) if total_trades else 0.0
        
        return {
            "total_trades": len(total_trades),
            "wins": tot_w,
            "losses": len(total_trades) - tot_w,
            "win_rate": wr,
            "total_net_pnl": tot_pnl,
            "avg_net_trade": avg,
            "active_sessions": active_sessions,
            "wr100_sessions": wr100_sessions,
            "profitable_sessions": profitable_sessions,
            "reasons": dict(reasons),
            "trades": total_trades,
            "cycle_summaries": cycle_summaries
        }

if __name__ == "__main__":
    baseline_eng = UnifiedPeakBenchmarkEngine()
    res = baseline_eng.evaluate_all_cycles()
    print("=== BASELINE (NO FIXES APPLIED) ===")
    print(f"Total Trades: {res['total_trades']} across {res['active_sessions']} sessions")
    print(f"Wins: {res['wins']} | Losses: {res['losses']} | Win Rate: {res['win_rate']:.2f}%")
    print(f"Total Net PnL: {res['total_net_pnl']:+.2f} bps | Avg: {res['avg_net_trade']:+.2f} bps/trade")
    print(f"100% WR Sessions: {res['wr100_sessions']}/{res['active_sessions']} ({res['wr100_sessions']/res['active_sessions']*100:.1f}%)")
    print(f"Profitable Sessions: {res['profitable_sessions']}/{res['active_sessions']} ({res['profitable_sessions']/res['active_sessions']*100:.1f}%)")
    print(f"Exit Reasons: {res['reasons']}")
