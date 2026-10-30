import json
import numpy as np

bars = [json.loads(l) for l in open('scratch/live_hierarchical_session_1h_ledger.jsonl')]

def run_sim(macro_support_filter=True):
    active_pos = None
    closed = []
    
    for i, b in enumerate(bars):
        mid = b['bbo_mid']
        bid = b['bbo_bid']
        ask = b['bbo_ask']
        ts_ms = b['ts_ms']
        
        p_mac = np.array(b['macro_probs'], dtype=np.float64)
        p_mes = np.array(b['meso_probs'], dtype=np.float64)
        p_tac = np.array(b['tactical_probs'], dtype=np.float64)
        p_mic = np.array(b['micro_probs'], dtype=np.float64)
        
        d_macro = float(p_mac[2] - p_mac[0])
        d_meso = float(p_mes[2] - p_mes[0])
        d_tact = float(p_tac[2] - p_tac[0])
        d_micro = float(p_mic[2] - p_mic[0])
        
        if active_pos is not None:
            side = active_pos['side']
            sign = 1.0 if side == "BUY" else -1.0
            entry_p = active_pos['entry_price']
            entry_ts = active_pos['entry_ts']
            peak_mfe = active_pos['peak_mfe']
            
            unrealized_bps = sign * ((mid - entry_p) / entry_p) * 1e4
            if unrealized_bps > peak_mfe:
                peak_mfe = unrealized_bps
                active_pos['peak_mfe'] = peak_mfe
                
            elapsed_sec = (ts_ms - entry_ts) / 1000.0
            
            # Check Full Take Profit (35 bps)
            if unrealized_bps >= 35.0:
                exit_p = entry_p * (1.0 + sign * 35.0 * 1e-4)
                net = 35.0 - 4.0
                closed.append((side, entry_p, exit_p, net, "RESTING_MAKER_TP", elapsed_sec))
                active_pos = None
            # Check Trailing Stop Lock
            elif peak_mfe >= 7.2:
                pullback_buf = max(3.2, 0.25 * peak_mfe)
                lock_bps = max(4.5, peak_mfe - pullback_buf)
                if unrealized_bps <= lock_bps:
                    exit_p = entry_p * (1.0 + sign * lock_bps * 1e-4)
                    net = lock_bps - 4.0
                    closed.append((side, entry_p, exit_p, net, "MICRO_TRAILING_MAKER_TP", elapsed_sec))
                    active_pos = None
            # Check Invalidation Exit
            elif elapsed_sec >= 45.0 and unrealized_bps < 0.0:
                if macro_support_filter:
                    # Invalidate only if Macro is not strongly supporting (sign * d_macro < 0.10)
                    # OR if Meso has suffered a major breakdown (sign * d_meso < -0.22)
                    macro_opposes = (sign * d_macro) < 0.10
                    severe_meso_breakdown = (sign * d_meso) < -0.22 and (sign * d_tact) < -0.05
                    sub_broken = (macro_opposes and (sign * d_meso) < -0.08 and (sign * d_tact) < -0.05) or severe_meso_breakdown
                else:
                    sub_broken = ((sign * d_meso) < -0.06 and (sign * d_tact) < -0.05)
                    
                if sub_broken:
                    exit_p = ask if side == "BUY" else bid
                    gross = sign * ((exit_p - entry_p) / entry_p) * 1e4
                    net = gross - 4.0
                    closed.append((side, entry_p, exit_p, net, "MACRO_REVERSAL_MAKER_EXIT", elapsed_sec))
                    active_pos = None
            # Check Hard Stop Loss (14.5 bps cap)
            elif unrealized_bps <= -14.5:
                exit_p = bid if side == "BUY" else ask
                gross = sign * ((exit_p - entry_p) / entry_p) * 1e4
                net = gross - 7.0
                closed.append((side, entry_p, exit_p, net, "HARD_STOP_LOSS_TAKER", elapsed_sec))
                active_pos = None
                
        # Check Entry
        if active_pos is None:
            if b.get('active_position') is not None and (i == 0 or bars[i-1].get('active_position') is None):
                ap = b['active_position']
                active_pos = {
                    'side': ap['side'],
                    'entry_price': ap['entry_price'],
                    'entry_ts': b['ts_ms'],
                    'peak_mfe': 0.0,
                }
                
    wins = sum(1 for c in closed if c[3] > 0)
    net_pnl = sum(c[3] for c in closed)
    print(f"\nmacro_support_filter = {macro_support_filter}:")
    print(f"  Closed Trades: {len(closed)} | Wins: {wins} | Net PnL: {net_pnl:+.2f} bps")
    for c in closed:
        print(f"    {c[0]} @ {c[1]:.2f} -> {c[2]:.2f} | net: {c[3]:+.2f} bps ({c[4]}) | dur: {c[5]:.0f}s")
    if active_pos is not None:
        p = active_pos
        sign = 1.0 if p['side'] == "BUY" else -1.0
        cur_net = sign * ((bars[-1]['bbo_mid'] - p['entry_price']) / p['entry_price']) * 1e4 - 4.0
        print(f"    Active at close: {p['side']} @ {p['entry_price']:.2f} (current net: {cur_net:+.2f} bps, dur: {(bars[-1]['ts_ms'] - p['entry_ts'])/1000:.0f}s)")

print("=== CYCLE 6 DATA ===")
run_sim(macro_support_filter=False)
run_sim(macro_support_filter=True)
