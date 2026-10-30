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

class UnifiedPeakReplayerWithDropAnalysis:
    def __init__(self, breaker_bps=45.0, breaker_is_taker=True):
        self.maker_fee = 2.0
        self.taker_fee = 4.5
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
        self.catastrophic_breaker = breaker_bps
        self.breaker_is_taker = breaker_is_taker
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
                    closed.append({"side": side, "net": net, "dur": dur, "mfe": mfe, "mae": pos["mae"], "reason": "TARGET_MAKER_TP", "entry_bar": pos["bar"], "exit_bar": idx})
                    cum_pnl += net
                    last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; last_exit_was_win = True; pos = None; continue
                # Ratchet Retrace Floor
                elif floor is not None and gross <= floor:
                    net = floor - (2 * self.maker_fee)
                    closed.append({"side": side, "net": net, "dur": dur, "mfe": mfe, "mae": pos["mae"], "reason": reason, "entry_bar": pos["bar"], "exit_bar": idx})
                    cum_pnl += net
                    last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; last_exit_was_win = True; pos = None; continue
                # Profitable Time Unwind
                elif dur >= self.max_dur and gross >= 4.0:
                    net = gross - (2 * self.maker_fee)
                    closed.append({"side": side, "net": net, "dur": dur, "mfe": mfe, "mae": pos["mae"], "reason": "TIME_UNWIND", "entry_bar": pos["bar"], "exit_bar": idx})
                    cum_pnl += net
                    last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; last_exit_was_win = True; pos = None; continue
                # Structural Circuit Breaker
                elif gross <= -self.catastrophic_breaker:
                    exit_fee = self.taker_fee if self.breaker_is_taker else self.maker_fee
                    # If market dropped beyond breaker in this bar, take actual adverse gross or breaker
                    actual_loss_gross = min(-self.catastrophic_breaker, gross)
                    net = actual_loss_gross - (self.maker_fee + exit_fee)
                    closed.append({"side": side, "net": net, "dur": dur, "mfe": mfe, "mae": pos["mae"], "reason": "CATASTROPHIC_BREAKER", "entry_bar": pos["bar"], "exit_bar": idx, "actual_gross": actual_loss_gross})
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
                
                side = None
                if p_dir == 2 and d_micro > 0.05 and d_meso > 0.0:
                    side = "BUY"
                elif p_dir == 0 and d_micro < -0.05 and d_meso < 0.0:
                    side = "SELL"
                elif d_tact > 0.15 and d_micro > 0.0:
                    side = "BUY"
                elif d_tact < -0.15 and d_micro < 0.0:
                    side = "SELL"
                    
                if edge < 5.0:
                    side = None
                    
                # Quantile Envelope Guard
                if side is not None and len(rolling_15m) >= 18:
                    s_hi = max(rolling_15m)
                    s_lo = min(rolling_15m)
                    rng = s_hi - s_lo
                    if rng >= 20.0:
                        pos_in_rng = (mid - s_lo) / rng
                        if side == "BUY" and pos_in_rng > 0.88:
                            side = None
                        elif side == "SELL" and pos_in_rng < 0.12:
                            side = None
                            
                # Consecutive loss cool-down
                if side is not None and not last_exit_was_win:
                    if (ts_ms - last_exit_ts) < 180000:
                        side = None
                        
                if side is not None:
                    entry_px = ask if side == "BUY" else bid
                    pos = {"side": side, "entry_px": entry_px, "ts": ts_ms, "mfe": 0.0, "mae": 0.0, "bar": idx}

        # Session end close
        if pos is not None:
            side = pos["side"]
            epx = pos["entry_px"]
            gross = (bid - epx)/epx*10000.0 if side == "BUY" else (epx - ask)/epx*10000.0
            net = gross - (2 * self.maker_fee)
            closed.append({"side": side, "net": net, "dur": (bars[-1].get("ts_ms", len(bars)*10000) - pos["ts"])/1000.0, "mfe": pos["mfe"], "mae": pos["mae"], "reason": "SESSION_END", "entry_bar": pos["bar"], "exit_bar": len(bars)-1})
            cum_pnl += net
            
        return closed, cum_pnl

if __name__ == "__main__":
    replayer_taker = UnifiedPeakReplayerWithDropAnalysis(breaker_bps=45.0, breaker_is_taker=True)
    replayer_maker = UnifiedPeakReplayerWithDropAnalysis(breaker_bps=45.0, breaker_is_taker=False)
    
    all_trades_taker = []
    all_trades_maker = []
    
    for d in dirs:
        p = os.path.join(base_dir, d)
        ledgers = glob.glob(os.path.join(p, "*ledger*.jsonl"))
        if not ledgers: continue
        t_tak, p_tak = replayer_taker.replay_ledger(ledgers[0])
        t_mak, p_mak = replayer_maker.replay_ledger(ledgers[0])
        for t in t_tak:
            t["cycle"] = d
            all_trades_taker.append(t)
        for t in t_mak:
            t["cycle"] = d
            all_trades_maker.append(t)
            
    print(f"=== REPLAY ANALYSIS OF SUDDEN DROPS & STOP LOSS ===")
    print(f"Total trades across 64 archives: {len(all_trades_taker)}")
    
    mae_list = [t["mae"] for t in all_trades_taker]
    print(f"Worst MAE encountered across all 64 cycles: {min(mae_list):.2f} bps")
    print(f"Median MAE: {sorted(mae_list)[len(mae_list)//2]:.2f} bps")
    print(f"75th percentile MAE: {sorted(mae_list)[int(len(mae_list)*0.25)]:.2f} bps")
    print(f"90th percentile MAE: {sorted(mae_list)[int(len(mae_list)*0.10)]:.2f} bps")
    
    print("\nAdverse Excursion (Drop) Depth Distribution:")
    print(f"  0.0 to -5.0 bps MAE (Virtually no drop): {sum(1 for m in mae_list if m >= -5.0)} ({sum(1 for m in mae_list if m >= -5.0)/len(all_trades_taker)*100:.1f}%)")
    print(f"  -5.0 to -15.0 bps MAE (Normal micro noise): {sum(1 for m in mae_list if -15.0 <= m < -5.0)} ({sum(1 for m in mae_list if -15.0 <= m < -5.0)/len(all_trades_taker)*100:.1f}%)")
    print(f"  -15.0 to -30.0 bps MAE (Intermediate adverse retrace): {sum(1 for m in mae_list if -30.0 <= m < -15.0)} ({sum(1 for m in mae_list if -30.0 <= m < -15.0)/len(all_trades_taker)*100:.1f}%)")
    print(f"  -30.0 to -45.0 bps MAE (Severe adverse drop, recovered): {sum(1 for m in mae_list if -45.0 < m < -30.0)} ({sum(1 for m in mae_list if -45.0 < m < -30.0)/len(all_trades_taker)*100:.1f}%)")
    print(f"  <= -45.0 bps MAE (Sudden Drop hitting Catastrophic Breaker): {sum(1 for m in mae_list if m <= -45.0)} ({sum(1 for m in mae_list if m <= -45.0)/len(all_trades_taker)*100:.1f}%)")
    
    breakers = [t for t in all_trades_taker if t["reason"] == "CATASTROPHIC_BREAKER"]
    print(f"\nTotal Sudden Drops Hitting Breaker: {len(breakers)}")
    for b in breakers:
        print(f"  Cycle: {b['cycle']:<18} Side: {b['side']} Net: {b['net']:>7.2f}b Dur: {b['dur']:>6.1f}s MAE: {b['mae']:>7.2f}b ExitGross: {b['actual_gross']:>7.2f}b")

    pnl_taker = sum(t["net"] for t in all_trades_taker)
    pnl_maker = sum(t["net"] for t in all_trades_maker)
    wins_taker = sum(1 for t in all_trades_taker if t["net"] > 0)
    
    print(f"\nOverall Performance Comparison:")
    print(f"  With Taker Emergency Stop (-45b + 4.5b taker): Win Rate = {wins_taker/len(all_trades_taker)*100:.2f}%, Total Net PnL = {pnl_taker:.2f} bps")
    print(f"  With Maker Emergency Stop (-45b + 2.0b maker): Win Rate = {wins_taker/len(all_trades_maker)*100:.2f}%, Total Net PnL = {pnl_maker:.2f} bps")
