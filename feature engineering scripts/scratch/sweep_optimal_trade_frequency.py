"""
feature engineering scripts/scratch/sweep_optimal_trade_frequency.py
Finds the mathematical peak frontier for trade count vs win rate vs net profitability.
"""
import json
from pathlib import Path
from collections import deque

archives = [
    (Path(r"D:\PrivateProjects\BTC Futures\scratch\archive_cycle10"), "Cycle 10"),
    (Path(r"D:\PrivateProjects\BTC Futures\scratch\archive_cycle11"), "Cycle 11"),
    (Path(r"D:\PrivateProjects\BTC Futures\scratch\archive_cycle12"), "Cycle 12"),
    (Path(r"D:\PrivateProjects\BTC Futures\scratch\archive_cycle13"), "Cycle 13"),
    (Path(r"D:\PrivateProjects\BTC Futures\scratch\archive_cycle14"), "Cycle 14"),
    (Path(r"D:\PrivateProjects\BTC Futures\scratch\archive_cycle15"), "Cycle 15"),
    (Path(r"D:\PrivateProjects\BTC Futures\scratch\archive_cycle16"), "Cycle 16"),
    (Path(r"D:\PrivateProjects\BTC Futures\scratch"), "Cycle 19"),
]

def load_all_sessions():
    sessions = []
    for arch, name in archives:
        files = list(arch.glob("*hierarchical_session_1h_ledger.jsonl"))
        if not files: continue
        with open(files[0], "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        sessions.append((name, rows))
    return sessions

def run_simulation(sessions, cooldown_win_s, cooldown_loss_s, min_edge, require_macro_neutral, min_meso_edge):
    all_trades = []
    for c_name, rows in sessions:
        recent_mids = deque(maxlen=36)
        trades = []
        active_pos = None
        last_exit_ts = 0
        last_exit_p = 0.0
        last_exit_dir = 1
        last_exit_was_win = True
        
        for r in rows:
            b = r['bar_idx']
            now_ms = r['ts_ms']
            mid = r['bbo_mid']
            bid = r.get('bbo_bid', mid - 0.05)
            ask = r.get('bbo_ask', mid + 0.05)
            
            d_macro = r['macro_probs'][2] - r['macro_probs'][0]
            d_meso = r['meso_probs'][2] - r['meso_probs'][0]
            d_tact = r['tactical_probs'][2] - r['tactical_probs'][0]
            d_micro = r['micro_probs'][2] - r['micro_probs'][0]
            
            # Position management
            if active_pos is not None:
                pos = active_pos
                sign = 1.0 if pos['side'] == 'BUY' else -1.0
                curr_exit_p = bid if pos['side'] == 'BUY' else ask
                unreal = sign * ((curr_exit_p - pos['entry_p']) / pos['entry_p']) * 1e4
                pos['mfe'] = max(pos['mfe'], unreal)
                mfe = pos['mfe']
                
                # Full TP
                if unreal >= 38.0:
                    trades.append({**pos, 'gross': unreal, 'net': unreal - 4.0, 'reason': 'FULL_TP'})
                    last_exit_ts = now_ms
                    last_exit_p = curr_exit_p
                    last_exit_dir = 2 if pos['side'] == 'BUY' else 0
                    last_exit_was_win = True
                    active_pos = None
                    recent_mids.append((now_ms, mid))
                    continue
                    
                # Invalidation Cut (Underwater)
                if unreal <= -7.0 and (d_meso * sign) < -0.12 and (d_tact * sign) < -0.12:
                    trades.append({**pos, 'gross': unreal, 'net': unreal - 7.0, 'reason': 'INVALIDATION'})
                    last_exit_ts = now_ms
                    last_exit_p = curr_exit_p
                    last_exit_dir = 2 if pos['side'] == 'BUY' else 0
                    last_exit_was_win = False
                    active_pos = None
                    recent_mids.append((now_ms, mid))
                    continue
                    
                # Hard SL
                if unreal <= -14.5:
                    trades.append({**pos, 'gross': unreal, 'net': unreal - 7.0, 'reason': 'HARD_SL'})
                    last_exit_ts = now_ms
                    last_exit_p = curr_exit_p
                    last_exit_dir = 2 if pos['side'] == 'BUY' else 0
                    last_exit_was_win = False
                    active_pos = None
                    recent_mids.append((now_ms, mid))
                    continue
                    
                # Trailing Profit Shield
                if mfe >= 4.50:
                    pos['armed'] = True
                    
                if pos['armed']:
                    if mfe < 7.0:
                        lock_bps = max(4.20, mfe - 1.00)
                    elif mfe < 10.0:
                        lock_bps = max(5.20, mfe - 1.50)
                    else:
                        lock_bps = max(7.50, mfe - 2.00)
                        
                    # Model Harvest
                    if unreal >= 4.80 and ((sign * d_micro) < -0.05 or (sign * d_tact) < -0.05):
                        trades.append({**pos, 'gross': unreal, 'net': unreal - 4.0, 'reason': 'MODEL_HARVEST'})
                        last_exit_ts = now_ms
                        last_exit_p = curr_exit_p
                        last_exit_dir = 2 if pos['side'] == 'BUY' else 0
                        last_exit_was_win = True
                        active_pos = None
                        recent_mids.append((now_ms, mid))
                        continue
                        
                    if unreal <= lock_bps:
                        exit_g = max(lock_bps, unreal)
                        trades.append({**pos, 'gross': exit_g, 'net': exit_g - 4.0, 'reason': 'TRAILING_FLOOR'})
                        last_exit_ts = now_ms
                        last_exit_p = curr_exit_p
                        last_exit_dir = 2 if pos['side'] == 'BUY' else 0
                        last_exit_was_win = True
                        active_pos = None
                        recent_mids.append((now_ms, mid))
                        continue
                        
            # Entry Evaluation
            if active_pos is None and b >= 15:
                setup_dir = None
                edge_val = r.get('expected_net_edge_bps', 0.0)
                
                # Check Macro alignment if required
                macro_ok_long = (d_macro >= -0.05) if require_macro_neutral else True
                macro_ok_short = (d_macro <= 0.05) if require_macro_neutral else True
                
                # Setup A: Meso Trend
                if d_meso >= min_meso_edge and d_tact > 0 and d_micro >= -0.02 and macro_ok_long:
                    setup_dir = 2
                elif d_meso <= -min_meso_edge and d_tact < 0 and d_micro <= 0.02 and macro_ok_short:
                    setup_dir = 0
                # Setup B: Strong Tactical Continuation aligned with Meso direction
                elif d_tact >= 0.20 and d_micro > 0.05 and d_meso >= 0.0 and macro_ok_long:
                    setup_dir = 2
                elif d_tact <= -0.20 and d_micro < -0.05 and d_meso <= 0.0 and macro_ok_short:
                    setup_dir = 0
                    
                if setup_dir is not None and edge_val >= min_edge:
                    can_enter = True
                    if last_exit_ts > 0 and setup_dir == last_exit_dir:
                        elapsed_s = (now_ms - last_exit_ts) / 1000.0
                        retrace_bps = (((last_exit_p - mid) / last_exit_p) * 1e4 if last_exit_dir == 2 else ((mid - last_exit_p) / last_exit_p) * 1e4)
                        req_cooldown = cooldown_win_s if last_exit_was_win else cooldown_loss_s
                        req_retrace = 5.0 if last_exit_was_win else 12.0
                        if elapsed_s < req_cooldown and retrace_bps < req_retrace:
                            can_enter = False
                            
                    if can_enter:
                        dir_mult = 1.0 if setup_dir == 2 else -1.0
                        p_30s = [p for ts, p in recent_mids if 20000 <= (now_ms - ts) <= 40000]
                        s_30s = dir_mult * ((mid - p_30s[0]) / p_30s[0] * 1e4) if p_30s else 0.0
                        
                        p_180s = [p for ts, p in recent_mids if 140000 <= (now_ms - ts) <= 200000]
                        s_180s = dir_mult * ((mid - p_180s[0]) / p_180s[0] * 1e4) if p_180s else 0.0
                        
                        p_300s = [p for ts, p in recent_mids if (now_ms - ts) >= 240000]
                        s_300s = dir_mult * ((mid - p_300s[0]) / p_300s[0] * 1e4) if p_300s else 0.0
                        
                        if not ((s_30s > 10.0) or (s_180s > 16.0) or (s_300s > 18.0)):
                            side = "BUY" if setup_dir == 2 else "SELL"
                            active_pos = {'entry_b': b, 'entry_p': bid if side == 'BUY' else ask, 'side': side, 'mfe': 0.0, 'armed': False}
                            
            recent_mids.append((now_ms, mid))
        all_trades.extend(trades)
    return all_trades

sessions = load_all_sessions()
print(f"Loaded {len(sessions)} historical 1-hour sessions.")
print(f"{'Config':32s} | {'Trades':6s} | {'Tr/Hr':5s} | {'Wins':5s} | {'Win Rate':8s} | {'Net PnL':10s} | {'Avg Net':8s} | {'PF':5s}")
print("-" * 95)

configs = [
    # (cooldown_win_s, cooldown_loss_s, min_edge, require_macro_neutral, min_meso_edge, label)
    (60, 180, 5.0, True, 0.08, "CD=60/180, Edge>=5, Meso>=0.08"),
    (90, 240, 5.0, True, 0.08, "CD=90/240, Edge>=5, Meso>=0.08"),
    (120, 240, 5.0, True, 0.08, "CD=120/240, Edge>=5, Meso>=0.08"),
    (180, 300, 5.0, True, 0.08, "CD=180/300, Edge>=5, Meso>=0.08"),
    (120, 240, 7.0, True, 0.08, "CD=120/240, Edge>=7, Meso>=0.08"),
    (120, 240, 5.0, True, 0.10, "CD=120/240, Edge>=5, Meso>=0.10"),
    (150, 300, 6.0, True, 0.09, "CD=150/300, Edge>=6, Meso>=0.09"),
    (180, 300, 7.0, True, 0.08, "CD=180/300, Edge>=7, Meso>=0.08"),
    (240, 360, 5.0, True, 0.08, "CD=240/360, Edge>=5, Meso>=0.08"),
    (300, 420, 5.0, True, 0.08, "CD=300/420, Edge>=5, Meso>=0.08"),
]

for cd_w, cd_l, edge, mac, meso, lbl in configs:
    tds = run_simulation(sessions, cd_w, cd_l, edge, mac, meso)
    n = len(tds)
    if n == 0: continue
    w = sum(1 for t in tds if t['net'] > 0)
    wr = w / n * 100.0
    net = sum(t['net'] for t in tds)
    avg_net = net / n
    gw = sum(t['net'] for t in tds if t['net'] > 0)
    gl = abs(sum(t['net'] for t in tds if t['net'] < 0))
    pf = (gw / gl) if gl > 0 else 999.0
    print(f"{lbl:32s} | {n:6d} | {n/len(sessions):5.1f} | {w:5d} | {wr:7.1f}% | {net:+9.2f}b | {avg_net:+7.2f}b | {pf:5.2f}")
