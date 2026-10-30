"""
feature engineering scripts/scratch/simulate_hierarchical_reentry.py
Simulates re-entry using the exact recorded HierarchicalDecisionOutput from the ledgers:
- Uses primary_direction in (0, 2)
- Uses execution_mode in ('PASSIVE_MAKER_SNIPER', 'URGENT_HYBRID', 'WAIT_FOR_MICRO_SNIPER')
- Uses expected_net_edge_bps >= min_edge
- Tests cooldowns from 60s to 360s
- With Multi-Horizon Impulse Exhaustion Guard
- With Model Harvest & Dynamic Trailing Shield
- With Dual-Horizon Invalidation Cut
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

def simulate(cooldown_sec, retrace_bps, min_edge_bps):
    all_trades = []
    for arch, name in archives:
        files = list(arch.glob("*hierarchical_session_1h_ledger.jsonl"))
        if not files: continue
        with open(files[0], "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
            
        recent_mids = deque(maxlen=36)
        trades = []
        active_pos = None
        last_exit_ts = 0
        last_exit_p = 0.0
        last_exit_dir = 1
        
        for r in rows:
            b = r['bar_idx']
            now_ms = r['ts_ms']
            mid = r['bbo_mid']
            bid = r.get('bbo_bid', mid - 0.05)
            ask = r.get('bbo_ask', mid + 0.05)
            pdir = r['primary_direction']
            edge = r['expected_net_edge_bps']
            mode = r['execution_mode']
            
            d_meso = r['meso_probs'][2] - r['meso_probs'][0]
            d_tact = r['tactical_probs'][2] - r['tactical_probs'][0]
            d_micro = r['micro_probs'][2] - r['micro_probs'][0]
            
            # Position management
            if active_pos is not None:
                pos = active_pos
                sign = 1.0 if pos['side'] == 'BUY' else -1.0
                opp_dir = 0 if pos['side'] == 'BUY' else 2
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
                    active_pos = None
                    recent_mids.append((now_ms, mid))
                    continue
                    
                # Invalidation Cut (Underwater)
                if unreal <= -7.0 and pdir == opp_dir and edge >= 12.0 and (d_meso * sign) < -0.15 and (d_tact * sign) < -0.15:
                    trades.append({**pos, 'gross': unreal, 'net': unreal - 7.0, 'reason': 'INVALIDATION'})
                    last_exit_ts = now_ms
                    last_exit_p = curr_exit_p
                    last_exit_dir = 2 if pos['side'] == 'BUY' else 0
                    active_pos = None
                    recent_mids.append((now_ms, mid))
                    continue
                    
                # Hard SL
                if unreal <= -14.5:
                    trades.append({**pos, 'gross': unreal, 'net': unreal - 7.0, 'reason': 'HARD_SL'})
                    last_exit_ts = now_ms
                    last_exit_p = curr_exit_p
                    last_exit_dir = 2 if pos['side'] == 'BUY' else 0
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
                    if unreal >= 4.80 and ((sign * d_micro) < -0.05 or pdir == opp_dir):
                        trades.append({**pos, 'gross': unreal, 'net': unreal - 4.0, 'reason': 'MODEL_HARVEST'})
                        last_exit_ts = now_ms
                        last_exit_p = curr_exit_p
                        last_exit_dir = 2 if pos['side'] == 'BUY' else 0
                        active_pos = None
                        recent_mids.append((now_ms, mid))
                        continue
                        
                    if unreal <= lock_bps:
                        exit_g = max(lock_bps, unreal)
                        trades.append({**pos, 'gross': exit_g, 'net': exit_g - 4.0, 'reason': 'TRAILING_FLOOR'})
                        last_exit_ts = now_ms
                        last_exit_p = curr_exit_p
                        last_exit_dir = 2 if pos['side'] == 'BUY' else 0
                        active_pos = None
                        recent_mids.append((now_ms, mid))
                        continue
                        
            # Entry Evaluation
            if active_pos is None and b >= 15:
                if pdir in (0, 2) and edge >= min_edge_bps and mode in ("PASSIVE_MAKER_SNIPER", "URGENT_HYBRID", "WAIT_FOR_MICRO_SNIPER"):
                    can_enter = True
                    if last_exit_ts > 0 and pdir == last_exit_dir:
                        elapsed_s = (now_ms - last_exit_ts) / 1000.0
                        retrace = (((last_exit_p - mid) / last_exit_p) * 1e4 if last_exit_dir == 2 else ((mid - last_exit_p) / last_exit_p) * 1e4)
                        if elapsed_s < cooldown_sec and retrace < retrace_bps:
                            can_enter = False
                            
                    if can_enter:
                        dir_mult = 1.0 if pdir == 2 else -1.0
                        p_30s = [p for ts, p in recent_mids if 20000 <= (now_ms - ts) <= 40000]
                        s_30s = dir_mult * ((mid - p_30s[0]) / p_30s[0] * 1e4) if p_30s else 0.0
                        
                        p_180s = [p for ts, p in recent_mids if 140000 <= (now_ms - ts) <= 200000]
                        s_180s = dir_mult * ((mid - p_180s[0]) / p_180s[0] * 1e4) if p_180s else 0.0
                        
                        p_300s = [p for ts, p in recent_mids if (now_ms - ts) >= 240000]
                        s_300s = dir_mult * ((mid - p_300s[0]) / p_300s[0] * 1e4) if p_300s else 0.0
                        
                        if not ((s_30s > 10.0) or (s_180s > 16.0) or (s_300s > 18.0)):
                            side = "BUY" if pdir == 2 else "SELL"
                            entry_p = bid if side == "BUY" else ask
                            active_pos = {'entry_b': b, 'entry_p': entry_p, 'side': side, 'mfe': 0.0, 'armed': False, 'cycle': name}
                            
            recent_mids.append((now_ms, mid))
        all_trades.extend(trades)
    return all_trades

print(f"{'Cooldown':12s} | {'Retrace':8s} | {'MinEdge':8s} | {'Trades':6s} | {'Tr/Hr':5s} | {'Wins':5s} | {'Win Rate':8s} | {'Net PnL':10s} | {'Avg Net':8s} | {'PF':5s}")
print("-" * 90)

for cd in [60, 120, 180, 240, 300, 360, 480]:
    for ret in [4.0, 6.0, 8.0, 12.0, 15.0]:
        for edge in [5.0, 7.0, 8.0]:
            tds = simulate(cd, ret, edge)
            n = len(tds)
            if n < 8: continue
            w = sum(1 for t in tds if t['net'] > 0)
            wr = w / n * 100.0
            net = sum(t['net'] for t in tds)
            avg_net = net / n
            gw = sum(t['net'] for t in tds if t['net'] > 0)
            gl = abs(sum(t['net'] for t in tds if t['net'] < 0))
            pf = (gw / gl) if gl > 0 else 999.0
            if wr >= 75.0 and net > 0:
                print(f"{cd:10d}s | {ret:6.1f}b | {edge:6.1f}b | {n:6d} | {n/8:5.1f} | {w:5d} | {wr:7.1f}% | {net:+9.2f}b | {avg_net:+7.2f}b | {pf:5.2f}")
