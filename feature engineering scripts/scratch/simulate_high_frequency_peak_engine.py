"""
feature engineering scripts/scratch/simulate_high_frequency_peak_engine.py
Simulates the High-Frequency Peak Engine across historical 1-hour sessions (Cycles 10-19):
1. Reduced Warmup: 10 bars (100s) instead of 36 bars (since REST bootstraps 100 bars)
2. Scaled Cooldown: 60s after WIN (retrace >= 4.0 bps), 120s after LOSS
3. Multi-Horizon Dual Authorization: Both Meso Swings and Tactical Impulses can trigger entries
4. Active Neural Model Profit Harvest: Closes profit at crest (gross >= 4.8 bps when micro weakens)
5. 3-Tier Dynamic Trailing Shield: Locks in floors at +4.2 bps (Tier 1), +5.2 bps (Tier 2), +7.5 bps (Tier 3)
6. Dual-Horizon Invalidation Cut: Underwater cut at -7.0 bps if tactical & meso flip
7. Multi-Horizon Impulse Exhaustion Guard: Blocks entries if 30s > 10 bps, 180s > 16 bps, 300s > 18 bps
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

def simulate_session(arch_path, cycle_name):
    ledger_files = list(arch_path.glob("*hierarchical_session_1h_ledger.jsonl"))
    if not ledger_files:
        return []
    with open(ledger_files[0], "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
        
    recent_mids = deque(maxlen=36)
    trades = []
    active_pos = None
    last_exit_ts = 0
    last_exit_p = 0.0
    last_exit_dir = 1
    last_exit_was_win = True
    
    for i, r in enumerate(rows):
        b = r['bar_idx']
        now_ms = r['ts_ms']
        mid = r['bbo_mid']
        bid = r.get('bbo_bid', mid - 0.05)
        ask = r.get('bbo_ask', mid + 0.05)
        
        d_macro = r['macro_probs'][2] - r['macro_probs'][0]
        d_meso = r['meso_probs'][2] - r['meso_probs'][0]
        d_tact = r['tactical_probs'][2] - r['tactical_probs'][0]
        d_micro = r['micro_probs'][2] - r['micro_probs'][0]
        
        # 1. Manage active position if open
        if active_pos is not None:
            pos = active_pos
            sign = 1.0 if pos['side'] == 'BUY' else -1.0
            opp_dir = 0 if pos['side'] == 'BUY' else 2
            curr_exit_p = bid if pos['side'] == 'BUY' else ask
            unreal = sign * ((curr_exit_p - pos['entry_p']) / pos['entry_p']) * 1e4
            pos['mfe'] = max(pos['mfe'], unreal)
            mfe = pos['mfe']
            
            # A. Full Take Profit (38 bps)
            if unreal >= 38.0:
                pnl = unreal - 4.0
                trades.append({**pos, 'exit_b': b, 'exit_p': curr_exit_p, 'gross': unreal, 'net': pnl, 'reason': 'FULL_TP'})
                last_exit_ts = now_ms
                last_exit_p = curr_exit_p
                last_exit_dir = 2 if pos['side'] == 'BUY' else 0
                last_exit_was_win = True
                active_pos = None
                recent_mids.append((now_ms, mid))
                continue
                
            # B. Dual-Horizon Invalidation Cut (Underwater)
            if unreal <= -7.0 and (d_meso * sign) < -0.12 and (d_tact * sign) < -0.12:
                pnl = unreal - 7.0  # Taker SL exit
                trades.append({**pos, 'exit_b': b, 'exit_p': curr_exit_p, 'gross': unreal, 'net': pnl, 'reason': 'DUAL_HORIZON_INVALIDATION'})
                last_exit_ts = now_ms
                last_exit_p = curr_exit_p
                last_exit_dir = 2 if pos['side'] == 'BUY' else 0
                last_exit_was_win = False
                active_pos = None
                recent_mids.append((now_ms, mid))
                continue
                
            # C. Hard Stop Loss (-14.5 bps)
            if unreal <= -14.5:
                pnl = unreal - 7.0
                trades.append({**pos, 'exit_b': b, 'exit_p': curr_exit_p, 'gross': unreal, 'net': pnl, 'reason': 'HARD_SL'})
                last_exit_ts = now_ms
                last_exit_p = curr_exit_p
                last_exit_dir = 2 if pos['side'] == 'BUY' else 0
                last_exit_was_win = False
                active_pos = None
                recent_mids.append((now_ms, mid))
                continue
                
            # D. Trailing Profit Shield & Active Model Harvest
            if mfe >= 4.50:
                pos['armed'] = True
                
            if pos['armed']:
                # Dynamic Trailing Floor
                if mfe < 7.0:
                    lock_bps = max(4.20, mfe - 1.00)
                elif mfe < 10.0:
                    lock_bps = max(5.20, mfe - 1.50)
                else:
                    lock_bps = max(7.50, mfe - 2.00)
                    
                # Active Neural Model Profit Harvest
                if unreal >= 4.80:
                    # If 10s micro stalls or turns against position
                    if (sign * d_micro) < -0.05 or (sign * d_tact) < -0.05:
                        pnl = unreal - 4.0
                        trades.append({**pos, 'exit_b': b, 'exit_p': curr_exit_p, 'gross': unreal, 'net': pnl, 'reason': 'MODEL_HARVEST'})
                        last_exit_ts = now_ms
                        last_exit_p = curr_exit_p
                        last_exit_dir = 2 if pos['side'] == 'BUY' else 0
                        last_exit_was_win = True
                        active_pos = None
                        recent_mids.append((now_ms, mid))
                        continue
                        
                if unreal <= lock_bps:
                    exit_g = max(lock_bps, unreal)
                    pnl = exit_g - 4.0
                    trades.append({**pos, 'exit_b': b, 'exit_p': curr_exit_p, 'gross': exit_g, 'net': pnl, 'reason': 'TRAILING_FLOOR'})
                    last_exit_ts = now_ms
                    last_exit_p = curr_exit_p
                    last_exit_dir = 2 if pos['side'] == 'BUY' else 0
                    last_exit_was_win = True
                    active_pos = None
                    recent_mids.append((now_ms, mid))
                    continue
                    
        # 2. Check for entry if IDLE and past initial warmup (10 bars = 100s)
        if active_pos is None and b >= 10:
            # Determine directional setup:
            # Setup A: Strong Meso Swing
            # Setup B: Strong Tactical Impulse with Micro alignment and no Meso conflict
            setup_dir = None
            if abs(d_meso) >= 0.08 and (d_meso * d_tact) > 0 and (d_meso * d_micro) >= -0.02:
                setup_dir = 2 if d_meso > 0 else 0
            elif abs(d_tact) >= 0.16 and (d_tact * d_micro) > 0.05 and (d_tact * d_meso) >= -0.03:
                setup_dir = 2 if d_tact > 0 else 0
                
            if setup_dir is not None:
                # Check Re-entry cooldown
                can_enter = True
                if last_exit_ts > 0 and setup_dir == last_exit_dir:
                    elapsed_s = (now_ms - last_exit_ts) / 1000.0
                    retrace_bps = (((last_exit_p - mid) / last_exit_p) * 1e4 if last_exit_dir == 2 else ((mid - last_exit_p) / last_exit_p) * 1e4)
                    
                    req_cooldown = 60.0 if last_exit_was_win else 120.0
                    req_retrace = 4.0 if last_exit_was_win else 10.0
                    
                    if elapsed_s < req_cooldown and retrace_bps < req_retrace:
                        can_enter = False
                        
                if can_enter:
                    # Check Multi-Horizon Impulse Exhaustion Guard
                    dir_mult = 1.0 if setup_dir == 2 else -1.0
                    p_30s = [p for ts, p in recent_mids if 20000 <= (now_ms - ts) <= 40000]
                    s_30s = dir_mult * ((mid - p_30s[0]) / p_30s[0] * 1e4) if p_30s else 0.0
                    
                    p_180s = [p for ts, p in recent_mids if 140000 <= (now_ms - ts) <= 200000]
                    s_180s = dir_mult * ((mid - p_180s[0]) / p_180s[0] * 1e4) if p_180s else 0.0
                    
                    p_300s = [p for ts, p in recent_mids if (now_ms - ts) >= 240000]
                    s_300s = dir_mult * ((mid - p_300s[0]) / p_300s[0] * 1e4) if p_300s else 0.0
                    
                    is_exhausted = (s_30s > 10.0) or (s_180s > 16.0) or (s_300s > 18.0)
                    
                    if not is_exhausted:
                        entry_side = "BUY" if setup_dir == 2 else "SELL"
                        entry_p = bid if entry_side == "BUY" else ask  # Maker BBO entry
                        active_pos = {
                            'cycle': cycle_name,
                            'entry_b': b,
                            'entry_p': entry_p,
                            'side': entry_side,
                            'mfe': 0.0,
                            'armed': False
                        }
                        
        recent_mids.append((now_ms, mid))
        
    return trades

print(f"{'='*110}")
print(f"HIGH-FREQUENCY PEAK TRADING ENGINE SIMULATION (CYCLES 10 TO 19)")
print(f"{'='*110}")
print(f"{'Cycle':10s} | {'Bar Range':11s} | {'Side':4s} | {'Entry P':8s} | {'Exit P':8s} | {'Gross':8s} | {'Net PnL':9s} | {'Exit Reason':25s}")
print(f"{'-'*110}")

all_trades = []
for arch, c_name in archives:
    c_trades = simulate_session(arch, c_name)
    all_trades.extend(c_trades)
    for t in c_trades:
        print(f"{t['cycle']:10s} | Bar {t['entry_b']:3d}->{t['exit_b']:3d} | {t['side']:4s} | {t['entry_p']:8.2f} | {t['exit_p']:8.2f} | {t['gross']:+7.2f}b | {t['net']:+8.2f}b | {t['reason']:25s}")

n = len(all_trades)
wins = sum(1 for t in all_trades if t['net'] > 0)
net_pnl = sum(t['net'] for t in all_trades)
gross_win = sum(t['net'] for t in all_trades if t['net'] > 0)
gross_loss = abs(sum(t['net'] for t in all_trades if t['net'] < 0))
pf = (gross_win / gross_loss) if gross_loss > 0 else 999.0

print(f"{'='*110}")
print(f"HIGH-FREQUENCY PEAK ENGINE OVERALL RESULTS:")
print(f"  Total Trades: {n} (Average: {n/len(archives):.1f} trades per 1-hour session)")
print(f"  Winning Trades: {wins} / {n}")
print(f"  Win Rate: {wins/n*100.0:.1f}%")
print(f"  Total Net Realized Cash Profit: {net_pnl:+.2f} bps")
print(f"  Profit Factor: {pf:.2f}")
print(f"  Average Net Profit per Trade: {net_pnl/n:+.2f} bps")
print(f"{'='*110}")
