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

from inference.live_v4_model_registry import LiveMultiHorizonV4ModelRegistry, _score_to_simplex
from fusion.hierarchical_fusion_v3 import HierarchicalMultiHorizonFusionEngine
from execution.hierarchical_sniper_engine import HierarchicalSniperExecutionEngine
from streaming.multi_horizon_ring_buffer import MultiHorizonLiveRingBuffer

def test_session(ledger_file, p_fl_threshold=0.55, edge_threshold=0.15):
    bars = []
    with open(ledger_file, 'r') as f:
        for line in f:
            bars.append(json.loads(line))

    rb = MultiHorizonLiveRingBuffer()
    reg = rb.model_registry
    fusion = HierarchicalMultiHorizonFusionEngine()
    engine = HierarchicalSniperExecutionEngine()

    orig_eval = fusion.evaluate_multi_horizon_tick

    def patched_eval(ts_ms, p_macro, p_meso, p_tactical, p_micro, atr_15m_bps, atr_1h_bps, atr_10s_bps, micro_safe_fill_prob, hmm_regime_probs=None):
        dec = orig_eval(
            ts_ms=ts_ms,
            p_macro=p_macro,
            p_meso=p_meso,
            p_tactical=p_tactical,
            p_micro=p_micro,
            atr_15m_bps=atr_15m_bps,
            atr_1h_bps=atr_1h_bps,
            atr_10s_bps=atr_10s_bps,
            micro_safe_fill_prob=micro_safe_fill_prob,
            hmm_regime_probs=hmm_regime_probs,
        )
        p_dn, p_fl, p_up = dec.fused_swing_probs
        edge = p_up - p_dn
        
        d_macro = float(p_macro[2] - p_macro[0])
        d_meso = float(p_meso[2] - p_meso[0])
        
        is_directional = abs(edge) >= edge_threshold and p_fl <= p_fl_threshold
        if (d_macro * d_meso) < -0.05:
            is_directional = False
            
        new_dir = 2 if (is_directional and edge > 0) else (0 if (is_directional and edge < 0) else 1)
        
        if new_dir in (0, 2):
            sign_target = 1.0 if new_dir == 2 else -1.0
            d_micro = float(p_micro[2] - p_micro[0])
            micro_agrees = (sign_target * d_micro) >= -0.05
            if micro_agrees and dec.expected_net_edge_bps >= 4.0:
                new_mode = "PASSIVE_MAKER_SNIPER"
            else:
                new_mode = "NONE"
        else:
            new_mode = "NONE"
            
        return dataclasses.replace(dec, primary_direction=new_dir, execution_mode=new_mode)

    fusion.evaluate_multi_horizon_tick = patched_eval

    for i, b in enumerate(bars):
        mid = b["bbo_mid"]
        bid = b["bbo_bid"]
        ask = b["bbo_ask"]
        ts_ms = b["ts_ms"]
        
        rb.push_10s_bar({
            "close": mid, "close_ts": ts_ms, "open": mid, "high": mid + 0.5, "low": mid - 0.5,
            "volume": 2.0, "buy_volume": 1.0, "sell_volume": 1.0, "bbo_bid_qty": 1.5, "bbo_ask_qty": 1.5,
        })
        bar_dict = {"bbo_bid_qty": 1.5, "bbo_ask_qty": 1.5, "buy_volume": 1.0, "sell_volume": 1.0}
        micro_p, safe_fill_prob = reg.predict_tier1_micro_and_survival(rb, bar_dict)

        dec = fusion.evaluate_multi_horizon_tick(
            ts_ms=ts_ms,
            p_macro=rb.cached_macro_probs,
            p_meso=rb.cached_meso_probs,
            p_tactical=rb.cached_tactical_probs,
            p_micro=micro_p,
            atr_15m_bps=20.0, atr_1h_bps=25.0, atr_10s_bps=3.0,
            micro_safe_fill_prob=safe_fill_prob,
            hmm_regime_probs=reg.cached_hmm_probs,
        )

        fsm_pre = engine.on_market_micro_tick(
            ts_ms=ts_ms, bbo_bid=bid, bbo_ask=ask, bbo_bid_qty=1.5, bbo_ask_qty=1.5,
            aggressive_sell_vol_at_bid=1.0, aggressive_buy_vol_at_ask=1.0, eth_1s_ret_bps=0.0,
            bar_high=mid + 0.5, bar_low=mid - 0.5, current_decision=dec,
        )
        fsm_post = engine.on_hierarchical_decision(
            decision=dec, bbo_bid=bid, bbo_ask=ask, bbo_bid_qty=1.5, bbo_ask_qty=1.5,
            equity_usdt=1000.0, intra_bar_open=mid,
        )

    closed = engine.closed_trades
    wins = sum(1 for t in closed if t.net_return_bps > 0)
    net_pnl = sum(t.net_return_bps for t in closed)
    print(f"\n--- {ledger_file} (p_fl<={p_fl_threshold}, edge>={edge_threshold}) ---")
    print(f"Closed Trades: {len(closed)} | Wins: {wins} | Net PnL: {net_pnl:+.2f} bps")
    for t in closed:
        print(f"  Trade: {t.side} @ {t.entry_price:.2f} -> {t.exit_price:.2f} | pnl: {t.net_return_bps:+.2f} bps ({t.exit_reason})")
    if engine.active_position:
        print(f"  Active: {engine.active_position.side} @ {engine.active_position.entry_price:.2f}")

test_session('scratch/live_hierarchical_session_1h_ledger.jsonl', p_fl_threshold=0.55, edge_threshold=0.15)
test_session('scratch/archive_cycle3/live_hierarchical_session_1h_ledger.jsonl', p_fl_threshold=0.55, edge_threshold=0.15)
