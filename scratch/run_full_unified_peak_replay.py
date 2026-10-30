import os
import glob
import json
import collections
import io
import sys

# Ensure UTF-8 output on Windows
if not getattr(sys.stdout, '_utf8_wrapped', False):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stdout._utf8_wrapped = True
    except Exception:
        pass

base_dir = "D:/PrivateProjects/BTC Futures/scratch"
archives = sorted([d for d in os.listdir(base_dir) if (d.startswith("archive_cycle") or d.startswith("archive_v3")) and os.path.isdir(os.path.join(base_dir, d))])

class UnifiedPeakExecutionEngineReplay:
    """
    Unified Peak Architecture Replay Engine:
    Integrates all historical peak innovations:
    1. Cycle 25/31: Same-Side Retrace Price Barrier ($10 retrace) + Break-Even Shield (+3.8b MFE -> +0.15b net)
    2. Cycle 35/60: Conformal Softmax Conviction + Alpha Target TP (+35.0b gross / +31.0b net)
    3. Cycle 43: Stepped Ratchets (Tier 1: +6.0b -> +2.25b net; Tier 2: +10.0b -> +4.50b net; Tier 3: +15.0b -> +8.50b net)
    4. Cycle 48: Tier 0.5 Crown Ratchet (+4.5b MFE -> +1.25b net cash floor)
    5. Quantile Range Guard: 0.25 <= Pos <= 0.75 in 15m range (eliminates top/bottom buying)
    6. Cycle 64: 100% Maker Post-Only execution at BBO (2.0 bps fee leg / 4.0 bps roundtrip, 0 taker penalty)
    7. Catastrophic Structural Breaker: -45.0 bps Maker limit exit
    """
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

    def replay_archive(self, arch_name):
        arch_path = os.path.join(base_dir, arch_name)
        ledgers = glob.glob(os.path.join(arch_path, "*ledger*.jsonl"))
        if not ledgers:
            return None
            
        with open(ledgers[0], "r", encoding="utf-8") as f:
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
            
            # 1. Manage Active Position (Universal IDLE Gate: completely decoupled from entry conditions)
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
                    
            # 2. Evaluate IDLE Entries
            if pos is None and idx >= 15:
                tact = bar.get("tactical_probs", [0.33, 0.33, 0.33])
                meso = bar.get("meso_probs", [0.33, 0.33, 0.33])
                micro = bar.get("micro_probs", [0.33, 0.33, 0.33])
                macro = bar.get("macro_probs", [0.33, 0.33, 0.33])
                p_dir = bar.get("primary_direction", bar.get("direction", 1))
                edge = bar.get("expected_net_edge_bps", bar.get("edge", 0.0))
                
                conf_tact = max(tact)
                conf_meso = max(meso)
                d_tact = tact[2] - tact[0]
                d_micro = micro[2] - micro[0]
                d_meso = meso[2] - meso[0]
                d_macro = macro[2] - macro[0]
                
                # Conformal conviction gate
                has_conviction = (conf_tact >= 0.42 and conf_meso >= 0.38) or (
                    abs(d_tact) >= 0.14 and abs(d_meso) >= 0.03
                )
                if not has_conviction:
                    continue
                    
                cand_dir = None
                tact_dir = 2 if d_tact >= 0.12 else (0 if d_tact <= -0.12 else 1)
                tact_confirmed = (tact_dir in (0, 2)) and (
                    (tact_dir == 2 and d_micro > 0.012 and d_meso >= 0.02) or
                    (tact_dir == 0 and d_micro < -0.012 and d_meso <= -0.02)
                )
                
                if tact_confirmed:
                    cand_dir = tact_dir
                elif p_dir in (0, 2) and (edge >= 28.0 or (edge >= 5.0 and edge <= 20.0)):
                    if p_dir == 2 and d_meso >= 0.02: cand_dir = 2
                    elif p_dir == 0 and d_meso <= -0.02: cand_dir = 0
                    
                if cand_dir in (0, 2):
                    # Quantile Range Envelope Filter (0.25 <= Pos <= 0.75)
                    if len(rolling_15m) >= 30:
                        hi = max(rolling_15m)
                        lo = min(rolling_15m)
                        rng = hi - lo
                        if rng >= 20.0:
                            pos_in_rng = (mid - lo) / rng
                            if cand_dir == 2 and pos_in_rng >= 0.75: continue
                            if cand_dir == 0 and pos_in_rng <= 0.25: continue
                            
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
            "archive": arch_name,
            "trades": cnt,
            "wins": wins,
            "losses": cnt - wins,
            "win_rate": wr,
            "net_pnl": cum_pnl,
            "avg_pnl": avg_pnl,
            "profit_factor": pf,
            "closed_trades": closed
        }

if __name__ == '__main__':
    engine = UnifiedPeakExecutionEngineReplay()
    all_results = []

    print("=" * 115)
    print("REPLAYING UNIFIED PEAK ARCHITECTURE (Cycle 48 Tier 0.5 + Cycle 43 Ratchets + Cycle 60 Runner + Quantile Guard)")
    print("=" * 115)
    print(f"{'Cycle Archive':<24} | {'Trades':<8} | {'Win Rate':<10} | {'Net Realized PnL':<18} | {'Avg/Trade':<14} | {'PF':<8}")
    print("-" * 115)

    for arch in archives:
        res = engine.replay_archive(arch)
        if res and res["trades"] > 0:
            all_results.append(res)
            print(f"{res['archive']:<24} | {res['trades']:<8} | {res['win_rate']:>5.1f}%    | {res['net_pnl']:>+12.2f} bps    | {res['avg_pnl']:>+8.2f} bps | {res['profit_factor']:>6.2f}")

    tot_trades = sum([r["trades"] for r in all_results])
    tot_wins = sum([r["wins"] for r in all_results])
    tot_losses = sum([r["losses"] for r in all_results])
    tot_net = sum([r["net_pnl"] for r in all_results])
    overall_wr = (tot_wins / tot_trades * 100.0) if tot_trades > 0 else 0.0
    overall_avg = (tot_net / tot_trades) if tot_trades > 0 else 0.0

    gross_win_all = sum([sum([t["net"] for t in r["closed_trades"] if t["net"] > 0]) for r in all_results])
    gross_loss_all = abs(sum([sum([t["net"] for t in r["closed_trades"] if t["net"] < 0]) for r in all_results]))
    overall_pf = (gross_win_all / max(1e-6, gross_loss_all)) if gross_loss_all > 0 else 999.0

    wr100_cnt = len([r for r in all_results if r["win_rate"] == 100.0])
    prof_cnt = len([r for r in all_results if r["net_pnl"] > 0.0])
    active_cnt = len(all_results)

    print("=" * 115)
    print(f"{'ALL-TIME REPLAY TOTALS':<24} | {tot_trades:<8} | {overall_wr:>5.1f}%    | {tot_net:>+12.2f} bps    | {overall_avg:>+8.2f} bps | {overall_pf:>6.2f}")
    print("=" * 115)
    print(f"\nTotal Active Cycles Replayed: {active_cnt} / {len(archives)}")
    print(f"Cycles with 100.0% Win Rate: {wr100_cnt} / {active_cnt} ({wr100_cnt/active_cnt*100.0:.1f}%)")
    print(f"Cycles strictly Net Profitable: {prof_cnt} / {active_cnt} ({prof_cnt/active_cnt*100.0:.1f}%)")

    # Exit reasons breakdown
    reasons = collections.Counter()
    for r in all_results:
        for t in r["closed_trades"]:
            reasons[t["reason"]] += 1

    print("\n--- EXIT MECHANISM DISTRIBUTION ---")
    for k, v in reasons.most_common():
        print(f"  {k:<32} : {v:>4} trades ({v/tot_trades*100.0:>5.1f}%)")

