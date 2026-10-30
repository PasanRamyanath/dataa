import json
from pathlib import Path
from collections import deque

archives = [
    (Path(r"D:\PrivateProjects\BTC Futures\scratch\archive_cycle10"), "cycle10"),
    (Path(r"D:\PrivateProjects\BTC Futures\scratch\archive_cycle11"), "cycle11"),
    (Path(r"D:\PrivateProjects\BTC Futures\scratch\archive_cycle12"), "cycle12"),
    (Path(r"D:\PrivateProjects\BTC Futures\scratch\archive_cycle13"), "cycle13"),
    (Path(r"D:\PrivateProjects\BTC Futures\scratch\archive_cycle14"), "cycle14"),
    (Path(r"D:\PrivateProjects\BTC Futures\scratch\archive_cycle15"), "cycle15"),
    (Path(r"D:\PrivateProjects\BTC Futures\scratch\archive_cycle16"), "cycle16"),
    (Path(r"D:\PrivateProjects\BTC Futures\scratch"), "cycle19"),
]

def simulate_full_session(arch, c_name):
    ledger_files = list(arch.glob("*hierarchical_session_1h_ledger.jsonl"))
    if not ledger_files:
        return []
    with open(ledger_files[0], "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
        
    recent_mids = deque(maxlen=36)
    max_30s = 10.0
    max_180s = 16.0
    max_300s = 18.0
    
    trades = []
    active_pos = None
    
    for i in range(len(rows)):
        r = rows[i]
        b = r['bar_idx']
        now_ms = r['ts_ms']
        mid = r['bbo_mid']
        pdir = r['primary_direction']
        edge = r['expected_net_edge_bps']
        mode = r['execution_mode']
        
        # 1. Update active position if one exists
        if active_pos is not None:
            pos = active_pos
            sign = 1.0 if pos['side'] == 'BUY' else -1.0
            opp_dir = 0 if pos['side'] == 'BUY' else 2
            unreal = sign * ((mid - pos['entry_p']) / pos['entry_p']) * 1e4
            pos['mfe'] = max(pos['mfe'], unreal)
            mfe = pos['mfe']
            
            # Check Hard TP
            if unreal >= 38.0:
                trades.append({**pos, 'exit_b': b, 'gross': 38.0, 'net': 34.0, 'reason': 'HARD_TP'})
                active_pos = None
                recent_mids.append((now_ms, mid))
                continue
                
            # Dual-Horizon Invalidation Cut (Underwater)
            d_meso = r['meso_probs'][2] - r['meso_probs'][0]
            d_tact = r['tactical_probs'][2] - r['tactical_probs'][0]
            if unreal <= -7.0 and pdir == opp_dir and edge >= 12.0 and (sign * d_meso) < -0.15 and (sign * d_tact) < -0.15:
                trades.append({**pos, 'exit_b': b, 'gross': unreal, 'net': unreal - 7.0, 'reason': 'DUAL_HORIZON_INVALIDATION'})
                active_pos = None
                recent_mids.append((now_ms, mid))
                continue
                
            # Hard SL
            if unreal <= -14.5:
                trades.append({**pos, 'exit_b': b, 'gross': -14.5, 'net': -21.5, 'reason': 'HARD_SL'})
                active_pos = None
                recent_mids.append((now_ms, mid))
                continue
                
            # Trailing Profit Shield
            if mfe >= 4.50:
                pos['armed'] = True
                
            if pos['armed']:
                if mfe < 7.0:
                    giveback = 1.0
                    lock_bps = max(4.20, mfe - giveback)
                elif mfe < 10.0:
                    giveback = 1.5
                    lock_bps = max(5.20, mfe - giveback)
                else:
                    giveback = 2.0
                    lock_bps = max(7.50, mfe - giveback)
                    
                # Active Model Harvest
                d_micro = r['micro_probs'][2] - r['micro_probs'][0]
                if unreal >= 4.80:
                    # Model detects momentum stalling or turning against trade
                    if (sign * d_micro) < -0.05 or pdir == opp_dir:
                        trades.append({**pos, 'exit_b': b, 'gross': unreal, 'net': unreal - 4.0, 'reason': 'MODEL_HARVEST'})
                        active_pos = None
                        recent_mids.append((now_ms, mid))
                        continue
                        
                if unreal <= lock_bps:
                    exit_g = max(lock_bps, unreal)
                    trades.append({**pos, 'exit_b': b, 'gross': exit_g, 'net': exit_g - 4.0, 'reason': 'TRAILING_FLOOR'})
                    active_pos = None
                    recent_mids.append((now_ms, mid))
                    continue
                    
        # 2. Check for new entry if flat and warm-up complete
        if active_pos is None and b >= 36:
            # Check if this bar had a historical entry attempt or trigger
            prev_ap = rows[i-1].get('active_position') if i > 0 else None
            curr_ap = r.get('active_position')
            
            # If historical engine entered on this bar:
            if prev_ap is None and curr_ap is not None:
                side = curr_ap['side']
                p_dir = 2 if side == 'BUY' else 0
                
                # Check Multi-Horizon Impulse Exhaustion Guard
                dir_mult = 1.0 if p_dir == 2 else -1.0
                p_30s = [p for ts, p in recent_mids if 20000 <= (now_ms - ts) <= 40000]
                s_30s = dir_mult * ((mid - p_30s[0]) / p_30s[0] * 1e4) if p_30s else 0.0
                
                p_180s = [p for ts, p in recent_mids if 140000 <= (now_ms - ts) <= 200000]
                s_180s = dir_mult * ((mid - p_180s[0]) / p_180s[0] * 1e4) if p_180s else 0.0
                
                p_300s = [p for ts, p in recent_mids if (now_ms - ts) >= 240000]
                s_300s = dir_mult * ((mid - p_300s[0]) / p_300s[0] * 1e4) if p_300s else 0.0
                
                is_exhausted = (s_30s > max_30s) or (s_180s > max_180s) or (s_300s > max_300s)
                
                if not is_exhausted:
                    active_pos = {
                        'entry_b': b,
                        'entry_p': curr_ap['entry_price'],
                        'side': side,
                        'mfe': 0.0,
                        'armed': False
                    }
                    
        recent_mids.append((now_ms, mid))
        
    return trades

print("=== COMBINED SIMULATION: EXHAUSTION GUARD + MODEL HARVEST ENGINE ===")
print(f"{'Cycle':10s} | {'Entry Bar':9s} | {'Side':4s} | {'Entry P':8s} | {'Exit P':8s} | {'Gross':8s} | {'Net PnL':9s} | {'Exit Reason':25s}")
print("-" * 105)

grand_total_net = 0.0
total_wins = 0
total_trades = 0

for arch, c_name in archives:
    res_trades = simulate_full_session(arch, c_name)
    for t in res_trades:
        total_trades += 1
        if t['net'] > 0:
            total_wins += 1
        grand_total_net += t['net']
        exit_p = t['entry_p'] * (1.0 + (1.0 if t['side'] == 'BUY' else -1.0) * t['gross'] * 1e-4)
        print(f"{c_name:10s} | Bar {t['entry_b']:3d}->{t['exit_b']:3d} | {t['side']:4s} | {t['entry_p']:8.2f} | {exit_p:8.2f} | {t['gross']:+7.2f}b | {t['net']:+8.2f}b | {t['reason']:25s}")

win_rate = (total_wins / total_trades * 100.0) if total_trades > 0 else 0.0
print("-" * 105)
print(f"COMBINED PEAK SYSTEM SUMMARY ACROSS ALL SESSIONS:")
print(f"  Total Trades: {total_trades}")
print(f"  Winning Trades: {total_wins}")
print(f"  Win Rate: {win_rate:.1f}%")
print(f"  Total Realized Net Profit: {grand_total_net:+.2f} bps")
