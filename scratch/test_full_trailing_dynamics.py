import json
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
FEAT_ENGINE_DIR = REPO_ROOT / "feature engineering scripts"
for p in [str(REPO_ROOT), str(FEAT_ENGINE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from execution.hierarchical_sniper_engine import HierarchicalSniperExecutionEngine, ClosedSwingTradeRecord
from fusion.hierarchical_fusion_v3 import HierarchicalDecisionOutput

def simulate_with_proper_trailing(ledger_path, max_tp=35.0, trail_pullback=3.0):
    bars = [json.loads(l) for l in open(ledger_path)]
    engine = HierarchicalSniperExecutionEngine(min_reentry_cooldown_sec=60)
    
    # Override engine's inner manage methods with proper trailing
    closed_records = []
    
    active_pos = None
    last_exit_ts = 0
    last_exit_dir = 1
    
    for i, b in enumerate(bars):
        mid = b["bbo_mid"]
        bid = b["bbo_bid"]
        ask = b["bbo_ask"]
        ts_ms = b["ts_ms"]
        
        p_mac = np.array(b['macro_probs'], dtype=np.float64)
        p_mes = np.array(b['meso_probs'], dtype=np.float64)
        p_tac = np.array(b['tactical_probs'], dtype=np.float64)
        p_mic = np.array(b['micro_probs'], dtype=np.float64)
        
        d_macro = float(p_mac[2] - p_mac[0])
        d_meso = float(p_mes[2] - p_mes[0])
        d_tact = float(p_tac[2] - p_tac[0])
        d_micro = float(p_mic[2] - p_mic[0])
        
        macro_conflict = (d_macro * d_meso < -0.01) and (abs(d_macro) > 0.05 and abs(d_meso) > 0.05)
        
        is_long = (i >= 6) and (d_macro >= -0.02) and (d_meso >= 0.10) and (d_tact >= -0.05) and not macro_conflict
        is_short = (i >= 6) and (d_macro <= 0.02) and (d_meso <= -0.10) and (d_tact <= 0.05) and not macro_conflict
        
        primary_dir = 2 if is_long else (0 if is_short else 1)
        swing_edge = d_meso if primary_dir != 1 else 0.0
        
        sign_target = 1.0 if primary_dir == 2 else (-1.0 if primary_dir == 0 else 0.0)
        coherence = sign_target * (0.25 * d_macro + 0.40 * d_meso + 0.25 * d_tact + 0.10 * d_micro)
        cross_alignment = float(np.clip(coherence * 1.85, -1.0, 1.0))
        
        meta_logit = -0.32 + 4.35 * abs(swing_edge) + 1.15 * cross_alignment
        meta_act_prob = float(np.clip(1.0 / (1.0 + np.exp(-meta_logit)), 0.02, 0.98))
        
        tp_bps = max_tp
        sl_bps = 14.0
        expected_gross_bps = meta_act_prob * tp_bps - (1.0 - meta_act_prob) * sl_bps
        expected_net_bps = expected_gross_bps - 4.0
        
        micro_agrees = (primary_dir in (0, 2)) and ((sign_target * d_micro) >= -0.03)
        sniper_prob = 0.65 if micro_agrees else 0.20
        
        # Position Management if in position
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
            
            # Check Full Take Profit
            if unrealized_bps >= tp_bps:
                exit_p = entry_p * (1.0 + sign * tp_bps * 1e-4)
                net = tp_bps - 4.0
                closed_records.append((side, entry_p, exit_p, net, "RESTING_MAKER_TP", elapsed_sec))
                last_exit_ts = ts_ms
                last_exit_dir = primary_dir
                active_pos = None
            # Check Trailing Stop Lock
            elif peak_mfe >= 6.5:
                # Trailing stop armed
                lock_bps = max(4.25, peak_mfe - trail_pullback)
                if unrealized_bps <= lock_bps:
                    exit_p = entry_p * (1.0 + sign * lock_bps * 1e-4)
                    net = lock_bps - 4.0
                    closed_records.append((side, entry_p, exit_p, net, "MICRO_TRAILING_MAKER_TP", elapsed_sec))
                    last_exit_ts = ts_ms
                    last_exit_dir = primary_dir
                    active_pos = None
            # Check Multi-Horizon Invalidation Exit
            elif elapsed_sec >= 45.0 and unrealized_bps < 0.0:
                if (sign * d_meso) < -0.06 and (sign * d_tact) < -0.05:
                    exit_p = ask if side == "BUY" else bid
                    gross = sign * ((exit_p - entry_p) / entry_p) * 1e4
                    net = gross - 4.0
                    closed_records.append((side, entry_p, exit_p, net, "MACRO_REVERSAL_MAKER_EXIT", elapsed_sec))
                    last_exit_ts = ts_ms
                    last_exit_dir = primary_dir
                    active_pos = None
            # Check Hard Stop Loss
            elif unrealized_bps <= -sl_bps:
                exit_p = bid if side == "BUY" else ask
                gross = sign * ((exit_p - entry_p) / entry_p) * 1e4
                net = gross - 7.0 # Taker fee
                closed_records.append((side, entry_p, exit_p, net, "HARD_STOP_LOSS_TAKER", elapsed_sec))
                last_exit_ts = ts_ms
                last_exit_dir = primary_dir
                active_pos = None
                
        # Check Entry Authorization if IDLE
        if active_pos is None:
            cooldown_ok = (ts_ms - last_exit_ts) >= 60000 or (primary_dir != last_exit_dir)
            if cooldown_ok and primary_dir in (0, 2) and meta_act_prob >= 0.50 and expected_net_bps >= 4.0 and sniper_prob >= 0.55:
                side = "BUY" if primary_dir == 2 else "SELL"
                entry_p = bid if side == "BUY" else ask
                active_pos = {
                    'side': side,
                    'entry_price': entry_p,
                    'entry_ts': ts_ms,
                    'peak_mfe': 0.0,
                }
                
    wins = sum(1 for r in closed_records if r[3] > 0)
    net_pnl = sum(r[3] for r in closed_records)
    wr = (wins / len(closed_records) * 100) if closed_records else 0.0
    print(f"\nSimulated (max_tp={max_tp}bps, trail_pullback={trail_pullback}bps):")
    print(f"  Closed Trades: {len(closed_records)} | Wins: {wins} | Win Rate: {wr:.1f}% | Net PnL: {net_pnl:+.2f} bps")
    for r in closed_records:
        print(f"    {r[0]} @ {r[1]:.2f} -> {r[2]:.2f} | net: {r[3]:+.2f} bps ({r[4]}) | dur: {r[5]:.0f}s")
    if active_pos is not None:
        print(f"  Active: {active_pos['side']} @ {active_pos['entry_price']:.2f}")

simulate_with_proper_trailing('scratch/live_hierarchical_session_1h_ledger.jsonl', max_tp=25.0, trail_pullback=3.0)
simulate_with_proper_trailing('scratch/live_hierarchical_session_1h_ledger.jsonl', max_tp=35.0, trail_pullback=4.0)
simulate_with_proper_trailing('scratch/live_hierarchical_session_1h_ledger.jsonl', max_tp=40.0, trail_pullback=5.0)
