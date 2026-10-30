import json
import sys
from pathlib import Path
import numpy as np
import dataclasses

REPO_ROOT = Path(__file__).resolve().parent.parent
FEAT_ENGINE_DIR = REPO_ROOT / "feature engineering scripts"
for p in [str(REPO_ROOT), str(FEAT_ENGINE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from execution.hierarchical_sniper_engine import HierarchicalSniperExecutionEngine, ClosedSwingTradeRecord

def test_engine_on_ledger(
    ledger_path,
    p_fl_thresh=0.52,
    min_edge_thresh=0.15,
    min_meso_conviction=0.10,
    reentry_cooldown_sec=60,
    enable_inval_exit=True
):
    bars = [json.loads(l) for l in open(ledger_path)]
    engine = HierarchicalSniperExecutionEngine(min_reentry_cooldown_sec=reentry_cooldown_sec)
    
    # Track metrics
    closed_trades = []
    
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
        
        # Macro-Meso alignment check
        macro_conflict = (d_macro * d_meso) < -0.05
        macro_gate = 0.45 if macro_conflict else 1.0
        
        raw_swing = (
            0.15 * p_mac
            + 0.45 * (p_mes * macro_gate)
            + 0.30 * p_tac
            + 0.10 * p_mic
        )
        analytical_swing = raw_swing / np.sum(raw_swing)
        p_dn, p_fl, p_up = float(analytical_swing[0]), float(analytical_swing[1]), float(analytical_swing[2])
        swing_edge = p_up - p_dn
        
        fast_impulse = 0.60 * d_tact + 0.40 * d_micro
        is_counter_impulse = (swing_edge * fast_impulse) < -0.025 and abs(fast_impulse) > 0.10
        
        is_directional = (
            abs(swing_edge) >= min_edge_thresh
            and p_fl <= p_fl_thresh
            and not is_counter_impulse
            and not macro_conflict
            and abs(d_meso) >= min_meso_conviction
        )
        
        if is_directional:
            primary_dir = 2 if swing_edge > 0 else 0
        else:
            primary_dir = 1
            
        sign_target = 1.0 if primary_dir == 2 else (-1.0 if primary_dir == 0 else 0.0)
        coherence = sign_target * (0.25 * d_macro + 0.40 * d_meso + 0.25 * d_tact + 0.10 * d_micro)
        cross_alignment = float(np.clip(coherence * 1.85, -1.0, 1.0))
        
        swing_conviction = abs(swing_edge) * (1.0 - 0.5 * p_fl)
        meta_logit = -0.32 + 4.35 * swing_conviction + 1.15 * cross_alignment
        meta_act_prob = float(np.clip(1.0 / (1.0 + np.exp(-meta_logit)), 0.02, 0.98))
        
        tp_bps = 35.0
        sl_bps = 12.0
        expected_gross_bps = meta_act_prob * tp_bps - (1.0 - meta_act_prob) * sl_bps
        expected_net_bps = expected_gross_bps - 4.0
        
        micro_agrees = (primary_dir in (0, 2)) and ((sign_target * d_micro) >= -0.03)
        sniper_prob = 0.65 if micro_agrees else 0.20
        
        if primary_dir in (0, 2) and meta_act_prob >= 0.52 and expected_net_bps >= 4.0 and sniper_prob >= 0.55:
            exec_mode = "PASSIVE_MAKER_SNIPER"
            final_size = 0.50
        else:
            exec_mode = "NONE"
            final_size = 0.0
            
        from fusion.hierarchical_fusion_v3 import HierarchicalDecisionOutput
        dec = HierarchicalDecisionOutput(
            ts_ms=ts_ms,
            macro_probs=(float(p_mac[0]), float(p_mac[1]), float(p_mac[2])),
            meso_probs=(float(p_mes[0]), float(p_mes[1]), float(p_mes[2])),
            tactical_probs=(float(p_tac[0]), float(p_tac[1]), float(p_tac[2])),
            micro_probs=(float(p_mic[0]), float(p_mic[1]), float(p_mic[2])),
            fused_swing_probs=(p_dn, p_fl, p_up),
            primary_direction=primary_dir,
            cross_horizon_alignment=round(cross_alignment, 4),
            expected_gross_move_bps=round(expected_gross_bps, 2),
            expected_net_edge_bps=round(expected_net_bps, 2),
            meta_act_probability=round(meta_act_prob, 4),
            micro_sniper_trigger_prob=sniper_prob,
            target_holding_seconds=1800,
            take_profit_bps=tp_bps,
            stop_loss_bps=sl_bps,
            kelly_position_fraction=final_size,
            execution_mode=exec_mode,
        )
        
        # Check Custom Invalidation Exit for active position before engine ticks
        if enable_inval_exit and engine.active_position is not None:
            pos = engine.active_position
            sign = 1.0 if pos.side == "BUY" else -1.0
            elapsed_sec = (ts_ms - pos.entry_ts_ms) / 1000.0
            unrealized_bps = sign * ((mid - pos.entry_price) / pos.entry_price) * 1e4
            # Invalidation trigger: underwater + multi-horizon trend flipped against pos
            if elapsed_sec >= 45.0 and unrealized_bps < 0.0:
                if (sign * d_meso) < -0.10 and (sign * d_tact) < -0.05:
                    exit_p = ask if pos.side == "BUY" else bid
                    engine._close_position(
                        exit_price=exit_p,
                        exit_ts_ms=ts_ms,
                        exit_fee_bps=2.0,
                        reason="MACRO_REVERSAL_MAKER_EXIT",
                    )
        
        engine.on_market_micro_tick(
            ts_ms=ts_ms, bbo_bid=bid, bbo_ask=ask, bbo_bid_qty=1.5, bbo_ask_qty=1.5,
            aggressive_sell_vol_at_bid=1.0, aggressive_buy_vol_at_ask=1.0, eth_1s_ret_bps=0.0,
            bar_high=mid + 0.5, bar_low=mid - 0.5, current_decision=dec,
        )
        engine.on_hierarchical_decision(
            decision=dec, bbo_bid=bid, bbo_ask=ask, bbo_bid_qty=1.5, bbo_ask_qty=1.5,
            equity_usdt=1000.0, intra_bar_open=mid,
        )
        
    closed = engine.closed_trades
    wins = sum(1 for t in closed if t.net_return_bps > 0)
    net_pnl = sum(t.net_return_bps for t in closed)
    print(f"\n{'='*60}")
    print(f"Results for {Path(ledger_path).name} (meso_conv>={min_meso_conviction}, cooldown={reentry_cooldown_sec}s, inval_exit={enable_inval_exit}):")
    print(f"Closed Trades: {len(closed)} | Wins: {wins} | Net PnL: {net_pnl:+.2f} bps | Win Rate: {(wins/len(closed)*100 if closed else 0):.1f}%")
    for t in closed:
        print(f"  {t.side} @ {t.entry_price:.2f} -> {t.exit_price:.2f} | pnl: {t.net_return_bps:+.2f} bps ({t.exit_reason}) | dur: {t.holding_duration_sec:.0f}s")
    if engine.active_position is not None:
        p = engine.active_position
        print(f"  Active: {p.side} @ {p.entry_price:.2f}")

test_engine_on_ledger('scratch/live_hierarchical_session_1h_ledger.jsonl', min_meso_conviction=0.10, reentry_cooldown_sec=60, enable_inval_exit=True)
test_engine_on_ledger('scratch/archive_cycle3/live_hierarchical_session_1h_ledger.jsonl', min_meso_conviction=0.10, reentry_cooldown_sec=60, enable_inval_exit=True)
