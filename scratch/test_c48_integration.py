import io
import sys
if not getattr(sys.stdout, '_utf8_wrapped', False):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stdout._utf8_wrapped = True

from fast_peak_optimizer import all_bars
import collections

def simulate_with_c48(shield_mfe, c48_hurdle, c48_floor, r1_mfe, r1_fl, target_tp, breaker_loss):
    total_trades = 0; total_wins = 0; total_net = 0.0; active_cycles = 0; wr100_cycles = 0; profitable_cycles = 0
    maker_fee = 2.0
    
    for arch, bars in all_bars:
        rolling_15m = collections.deque(maxlen=90)
        pos = None; closed = []; cum_pnl = 0.0
        last_exit_ts = 0; last_exit_side = ""; last_exit_px = 0.0; last_exit_was_win = True
        
        for idx, bar in enumerate(bars):
            mid = bar.get("bbo_mid", bar.get("mid", 0.0))
            bid = bar.get("bbo_bid", mid - 0.05); ask = bar.get("bbo_ask", mid + 0.05)
            ts_ms = bar.get("ts_ms", idx * 10000)
            rolling_15m.append(mid)
            
            if pos is not None:
                side = pos["side"]; epx = pos["entry_px"]
                gross = (bid - epx)/epx*10000.0 if side == "BUY" else (epx - ask)/epx*10000.0
                pos["mfe"] = max(pos["mfe"], gross); pos["mae"] = min(pos["mae"], gross)
                mfe = pos["mfe"]; dur = (ts_ms - pos["ts"]) / 1000.0
                
                floor = None
                if mfe >= r1_mfe: floor = r1_fl
                elif c48_hurdle is not None and mfe >= c48_hurdle: floor = c48_floor
                elif mfe >= shield_mfe: floor = 4.15 # +0.15b net
                
                if gross >= target_tp:
                    net = gross - 4.0; closed.append(net); cum_pnl += net
                    last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; last_exit_was_win = True; pos = None; continue
                elif floor is not None and gross <= floor:
                    net = floor - 4.0; closed.append(net); cum_pnl += net
                    last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; last_exit_was_win = True; pos = None; continue
                elif dur >= 900.0 and gross >= 4.0:
                    net = gross - 4.0; closed.append(net); cum_pnl += net
                    last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; last_exit_was_win = True; pos = None; continue
                elif breaker_loss is not None and gross <= -breaker_loss:
                    net = -breaker_loss - 4.0; closed.append(net); cum_pnl += net
                    last_exit_ts = ts_ms; last_exit_side = side; last_exit_px = mid; last_exit_was_win = False; pos = None; continue
                    
            if pos is None and idx >= 15:
                tact = bar.get("tactical_probs", [0.33, 0.33, 0.33])
                meso = bar.get("meso_probs", [0.33, 0.33, 0.33])
                micro = bar.get("micro_probs", [0.33, 0.33, 0.33])
                p_dir = bar.get("primary_direction", bar.get("direction", 1))
                edge = bar.get("expected_net_edge_bps", bar.get("edge", 0.0))
                
                conf_tact = max(tact); conf_meso = max(meso)
                d_tact = tact[2] - tact[0]; d_micro = micro[2] - micro[0]; d_meso = meso[2] - meso[0]
                
                has_conviction = (conf_tact >= 0.42 and conf_meso >= 0.38) or (abs(d_tact) >= 0.14 and abs(d_meso) >= 0.03)
                if not has_conviction: continue
                
                cand_dir = None
                tact_dir = 2 if d_tact >= 0.12 else (0 if d_tact <= -0.12 else 1)
                tact_confirmed = (tact_dir in (0, 2)) and (
                    (tact_dir == 2 and d_micro > 0.012 and d_meso >= 0.02) or
                    (tact_dir == 0 and d_micro < -0.012 and d_meso <= -0.02)
                )
                if tact_confirmed: cand_dir = tact_dir
                elif p_dir in (0, 2) and (edge >= 28.0 or (edge >= 5.0 and edge <= 20.0)):
                    if p_dir == 2 and d_meso >= 0.02: cand_dir = 2
                    elif p_dir == 0 and d_meso <= -0.02: cand_dir = 0
                    
                if cand_dir in (0, 2):
                    if len(rolling_15m) >= 30:
                        hi = max(rolling_15m); lo = min(rolling_15m); rng = hi - lo
                        if rng >= 20.0:
                            pos_in_rng = (mid - lo) / rng
                            if cand_dir == 2 and pos_in_rng >= 0.75: continue
                            if cand_dir == 0 and pos_in_rng <= 0.25: continue
                            
                    if last_exit_ts > 0:
                        elapsed = (ts_ms - last_exit_ts) / 1000.0
                        if not last_exit_was_win and elapsed < 120.0: continue
                        elif last_exit_was_win and elapsed < 15.0: continue
                        if elapsed < 200.0 and last_exit_px > 0:
                            if cand_dir == 2 and last_exit_side == "BUY" and mid > (last_exit_px - 10.0): continue
                            if cand_dir == 0 and last_exit_side == "SELL" and mid < (last_exit_px + 10.0): continue
                    pos = {"side": "BUY" if cand_dir == 2 else "SELL", "entry_px": ask if cand_dir == 2 else bid, "ts": ts_ms, "mfe": 0.0, "mae": 0.0}
                    
        if closed:
            active_cycles += 1; w = len([c for c in closed if c > 0])
            total_trades += len(closed); total_wins += w; total_net += cum_pnl
            if cum_pnl > 0: profitable_cycles += 1
            if w == len(closed): wr100_cycles += 1
            
    wr = (total_wins / total_trades * 100.0) if total_trades > 0 else 0.0
    avg = (total_net / total_trades) if total_trades > 0 else 0.0
    return {
        "trades": total_trades, "wins": total_wins, "losses": total_trades - total_wins,
        "wr": wr, "net_pnl": total_net, "avg": avg, "active_cycles": active_cycles,
        "wr100_cycles": wr100_cycles, "profitable_cycles": profitable_cycles
    }

print("Running Systematic Comparison:")
# 1. Base without C48
r_base = simulate_with_c48(shield_mfe=5.0, c48_hurdle=None, c48_floor=None, r1_mfe=6.0, r1_fl=6.25, target_tp=35.0, breaker_loss=45.0)
print(f"Without C48 Tier 0.5: Tr: {r_base['trades']} | WR: {r_base['wr']:.1f}% | Net: {r_base['net_pnl']:+.2f} bps | Avg: {r_base['avg']:+.2f} bps | 100% WR: {r_base['wr100_cycles']}/{r_base['active_cycles']} | Profitable: {r_base['profitable_cycles']}/{r_base['active_cycles']}")

# 2. With C48 Tier 0.5 at +4.5b hurdle (locking +1.25b net floor)
r_c48 = simulate_with_c48(shield_mfe=5.0, c48_hurdle=4.5, c48_floor=5.25, r1_mfe=6.0, r1_fl=6.25, target_tp=35.0, breaker_loss=45.0)
print(f"With C48 Tier 0.5 (+4.5b): Tr: {r_c48['trades']} | WR: {r_c48['wr']:.1f}% | Net: {r_c48['net_pnl']:+.2f} bps | Avg: {r_c48['avg']:+.2f} bps | 100% WR: {r_c48['wr100_cycles']}/{r_c48['active_cycles']} | Profitable: {r_c48['profitable_cycles']}/{r_c48['active_cycles']}")
