import json
import os

class MockTrade:
    def __init__(self, side, entry_price, entry_ts, entry_bar):
        self.side = side
        self.entry_price = entry_price
        self.entry_ts = entry_ts
        self.entry_bar = entry_bar
        self.peak_mfe = 0.0
        self.max_mae = 0.0

def simulate_exact_cycle48(cycle_id):
    fpath = f"scratch/archive_cycle{cycle_id}/live_hierarchical_session_1h_ledger.jsonl"
    if not os.path.exists(fpath):
        return None
    
    bars = []
    with open(fpath, 'r', encoding='utf-8') as f:
        for line in f:
            bars.append(json.loads(line))
            
    rolling_15m = []
    recent_mids = []
    closed_trades = []
    active_pos = None
    last_exit_ts = 0
    last_exit_win = True
    last_exit_price = 0.0
    
    for i, b in enumerate(bars):
        mid = b['bbo_mid']
        bid = b['bbo_bid']
        ask = b['bbo_ask']
        ts = b['ts_ms']
        rolling_15m.append(mid)
        recent_mids.append((ts, mid))
        if len(rolling_15m) > 90:
            rolling_15m.pop(0)
            
        t_probs = b['tactical_probs']
        m_probs = b['meso_probs']
        mic_probs = b['micro_probs']
        mac_probs = b['macro_probs']
        
        d_tact = t_probs[2] - t_probs[0]
        d_meso = m_probs[2] - m_probs[0]
        d_mic = mic_probs[2] - mic_probs[0]
        d_mac = mac_probs[2] - mac_probs[0]
        
        # Position Management
        if active_pos is not None:
            unreal_bps = ((mid - active_pos.entry_price) / active_pos.entry_price) * 1e4
            if unreal_bps > active_pos.peak_mfe:
                active_pos.peak_mfe = unreal_bps
            if unreal_bps < active_pos.max_mae:
                active_pos.max_mae = unreal_bps
                
            holding_sec = (ts - active_pos.entry_ts) / 1000.0
            
            exit_trade = False
            exit_price = mid
            exit_reason = ""
            
            # Dual-Tier Stepped Ratchet
            mfe = active_pos.peak_mfe
            floor_gross = None
            if mfe >= 15.0:
                floor_gross = 12.5
            elif mfe >= 10.0:
                floor_gross = 8.5
            elif mfe >= 6.0:
                floor_gross = 6.0
            elif mfe >= 4.5:
                floor_gross = 5.25
                
            if floor_gross is not None and unreal_bps <= floor_gross:
                exit_price = active_pos.entry_price * (1.0 + floor_gross * 1e-4)
                exit_reason = "DUAL_TIER_RATCHET_TP"
                exit_trade = True
            elif d_meso < -0.08 or d_tact < -0.10:
                exit_price = bid
                exit_reason = "RESPONSIVE_MODEL_MAKER_EXIT"
                exit_trade = True
            elif holding_sec >= 300.0:
                exit_price = bid
                exit_reason = "PASSIVE_TIME_UNWIND"
                exit_trade = True
            elif unreal_bps <= -14.5:
                exit_price = bid
                exit_reason = "STOP_LOSS"
                exit_trade = True
                
            if exit_trade:
                gross_bps = ((exit_price - active_pos.entry_price) / active_pos.entry_price) * 1e4
                fee_bps = 4.0
                net_bps = gross_bps - fee_bps
                closed_trades.append({
                    "entry_bar": active_pos.entry_bar,
                    "exit_bar": b['bar_idx'],
                    "entry_price": active_pos.entry_price,
                    "exit_price": exit_price,
                    "gross_bps": gross_bps,
                    "net_bps": net_bps,
                    "holding_sec": holding_sec,
                    "peak_mfe": active_pos.peak_mfe,
                    "max_mae": active_pos.max_mae,
                    "reason": exit_reason,
                })
                last_exit_ts = ts
                last_exit_win = (net_bps > 0)
                last_exit_price = exit_price
                active_pos = None
                continue
                
        # Exact Cycle 48 Entry Logic
        if active_pos is None and i >= 10:
            # 1. Conformal certainty
            conf_tact = max(t_probs)
            conf_meso = max(m_probs)
            if conf_tact < 0.45 or conf_meso < 0.45:
                continue
                
            # 2. Tactical direction & confirmation
            tact_dir = 2 if d_tact >= 0.14 else (0 if d_tact <= -0.14 else 1)
            tact_confirmed = (tact_dir == 2 and d_mic > 0.02 and d_meso >= 0.04)
            if not tact_confirmed:
                continue
                
            # 3. Quad-Horizon long agreement
            quad_long = (d_mic > 0) and (d_tact > 0) and (d_meso > 0) and (d_mac > -0.05)
            if not quad_long:
                continue
                
            # 4. Range boundary violation: veto if mid > 15m high - $10.0
            if len(rolling_15m) >= 30:
                max_15m = max(rolling_15m)
                if mid > max_15m - 10.0:
                    continue
                    
            # 5. Parabolic surge: veto if fast surge
            past_30s = [p for t_old, p in recent_mids if (ts - t_old) <= 30000]
            if len(past_30s) >= 2:
                fast_ret_bps = ((mid - past_30s[0]) / past_30s[0]) * 1e4
                if fast_ret_bps > 10.0:
                    continue
                    
            # 6. Retrace cooldown
            if last_exit_ts > 0:
                elapsed = (ts - last_exit_ts) / 1000.0
                if not last_exit_win and elapsed < 60.0:
                    continue
                elif last_exit_win and elapsed < 15.0:
                    continue
                if elapsed < 200.0 and last_exit_price > 0:
                    if mid > last_exit_price - 10.0:
                        continue
                        
            # Enter Long @ Best Bid (100% Maker)
            active_pos = MockTrade(side="BUY", entry_price=bid, entry_ts=ts, entry_bar=b['bar_idx'])
            
    n_trades = len(closed_trades)
    wins = sum(1 for t in closed_trades if t['net_bps'] > 0)
    wr = (wins / n_trades * 100.0) if n_trades > 0 else 0.0
    net_pnl = sum(t['net_bps'] for t in closed_trades)
    return {
        "cycle": cycle_id,
        "trades": n_trades,
        "wins": wins,
        "losses": n_trades - wins,
        "win_rate": wr,
        "net_pnl": net_pnl,
        "trade_list": closed_trades
    }

print("Simulating EXACT Cycle 48 Configuration across Landmark Cycles:")
for c in [35, 42, 43, 48, 54, 55]:
    res = simulate_exact_cycle48(c)
    if res:
        print(f"Cycle {c:02d} | Trades: {res['trades']:2d} | Wins: {res['wins']:2d} | Losses: {res['losses']:2d} | WR: {res['win_rate']:5.1f}% | Net PnL: {res['net_pnl']:+6.2f} bps")
        for t in res['trade_list']:
            print(f"   Bar {t['entry_bar']} -> Bar {t['exit_bar']} ({t['holding_sec']:.0f}s) | Entry: {t['entry_price']} | Exit: {t['exit_price']:.2f} | Net: {t['net_bps']:+5.2f} bps | MFE: {t['peak_mfe']:.2f}b | Reason: {t['reason']}")
