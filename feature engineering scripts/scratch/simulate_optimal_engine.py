import json
from pathlib import Path

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

def simulate_trade(entry_idx, entry_p, side, rows, config):
    sign = 1.0 if side == "BUY" else -1.0
    opp_dir = 0 if side == "BUY" else 2
    peak_mfe = 0.0
    armed = False
    
    for i in range(entry_idx, len(rows)):
        r = rows[i]
        b = r['bar_idx']
        mid = r['bbo_mid']
        unreal = sign * ((mid - entry_p) / entry_p) * 1e4
        peak_mfe = max(peak_mfe, unreal)
        
        # Check Hard TP
        if unreal >= 38.0:
            return {'exit_bar': b, 'gross': 38.0, 'net': 34.0, 'reason': 'HARD_TP', 'mfe': peak_mfe}
            
        # Strategy B: High-Confidence Dual-Horizon Invalidation Cut (Underwater)
        if config.get('use_invalidation_cut', False):
            d_meso = r['meso_probs'][2] - r['meso_probs'][0]
            d_tact = r['tactical_probs'][2] - r['tactical_probs'][0]
            pdir = r['primary_direction']
            edge = r['expected_net_edge_bps']
            # ONLY cut if loss is beyond normal noise (unreal <= -7.0 bps) AND both Meso and Tactical confirmed opposite
            if unreal <= -7.0 and pdir == opp_dir and edge >= 12.0 and (sign * d_meso) < -0.15 and (sign * d_tact) < -0.15:
                net = unreal - 7.0 # conservative taker fee
                return {'exit_bar': b, 'gross': unreal, 'net': net, 'reason': 'DUAL_HORIZON_INVALIDATION', 'mfe': peak_mfe}

        # Check Hard SL
        if unreal <= -14.5:
            return {'exit_bar': b, 'gross': -14.5, 'net': -21.5, 'reason': 'HARD_SL', 'mfe': peak_mfe}
            
        # Strategy A: Dynamic Trailing Floor & Model Harvest
        if peak_mfe >= 4.50:
            armed = True
            
        if armed:
            if peak_mfe < 7.0:
                # Tier 1: Peak between 4.5 and 7.0
                giveback = config.get('tier1_giveback', 1.0)
                lock_bps = max(4.20, peak_mfe - giveback)
            elif peak_mfe < 10.0:
                # Tier 2: Peak between 7.0 and 10.0
                giveback = config.get('tier2_giveback', 1.5)
                lock_bps = max(5.20, peak_mfe - giveback)
            else:
                # Tier 3: Peak >= 10.0
                giveback = config.get('tier3_giveback', 2.0)
                lock_bps = max(7.50, peak_mfe - giveback)
                
            # Active Model Harvest: If in profit >= 5.0 bps and micro momentum flips against trade
            d_micro = r['micro_probs'][2] - r['micro_probs'][0]
            pdir = r['primary_direction']
            if config.get('use_model_harvest', True) and unreal >= 4.80:
                if (sign * d_micro) < -0.05 or pdir == opp_dir:
                    net = unreal - 4.0
                    return {'exit_bar': b, 'gross': unreal, 'net': net, 'reason': 'MODEL_HARVEST', 'mfe': peak_mfe}
                    
            if unreal <= lock_bps:
                exit_gross = max(lock_bps, unreal)
                net = exit_gross - 4.0
                return {'exit_bar': b, 'gross': exit_gross, 'net': net, 'reason': 'TRAILING_FLOOR', 'mfe': peak_mfe}
                
    last_unreal = sign * ((rows[-1]['bbo_mid'] - entry_p) / entry_p) * 1e4
    return {'exit_bar': rows[-1]['bar_idx'], 'gross': last_unreal, 'net': last_unreal - 4.0, 'reason': 'SESSION_END', 'mfe': peak_mfe}

print("=== RUNNING FULL HISTORICAL SIMULATION (CYCLES 10 TO 19) ===")
print(f"{'Cycle':10s} | {'Bar':3s} | {'Side':4s} | {'Entry':8s} | {'Peak MFE':8s} | {'Actual Net':10s} | {'Upgraded Net':12s} | {'Delta':8s} | {'Exit Reason':25s}")
print("-" * 105)

total_actual_net = 0.0
total_upgraded_net = 0.0

config_upgraded = {
    'tier1_giveback': 1.0,
    'tier2_giveback': 1.5,
    'tier3_giveback': 2.0,
    'use_model_harvest': True,
    'use_invalidation_cut': True
}

for arch, c_name in archives:
    ledger_files = list(arch.glob("*hierarchical_session_1h_ledger.jsonl"))
    metrics_files = list(arch.glob("*final_metrics.json"))
    if not ledger_files or not metrics_files:
        continue
    with open(ledger_files[0], "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    with open(metrics_files[0], "r", encoding="utf-8") as f:
        metrics = json.load(f)
    closed = metrics.get('closed_trades', [])
    
    for t in closed:
        entry_p = t['entry_price']
        side = t['side']
        actual_net = t['net_return_bps']
        
        entry_idx = -1
        for i in range(1, len(rows)):
            prev_ap = rows[i-1].get('active_position')
            curr_ap = rows[i].get('active_position')
            if prev_ap is None and curr_ap is not None:
                if curr_ap['side'] == side and abs(curr_ap['entry_price'] - entry_p) < 0.1:
                    entry_idx = i
                    break
        if entry_idx == -1:
            continue
            
        res = simulate_trade(entry_idx, entry_p, side, rows, config_upgraded)
        delta = res['net'] - actual_net
        total_actual_net += actual_net
        total_upgraded_net += res['net']
        
        print(f"{c_name:10s} | {rows[entry_idx]['bar_idx']:3d} | {side:4s} | {entry_p:8.2f} | {res['mfe']:+7.2f}b | {actual_net:+9.2f}b | {res['net']:+10.2f}b | {delta:+7.2f}b | {res['reason']:25s}")

print("-" * 105)
print(f"TOTAL REALIZED NET PNL ACROSS ALL CYCLES:")
print(f"  ACTUAL BASELINE NET PNL: {total_actual_net:+.2f} bps")
print(f"  UPGRADED ENGINE NET PNL: {total_upgraded_net:+.2f} bps (IMPROVEMENT: {total_upgraded_net - total_actual_net:+.2f} bps!)")
