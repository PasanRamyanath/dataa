import json
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
FEAT_ENGINE_DIR = REPO_ROOT / "feature engineering scripts"
for p in [str(REPO_ROOT), str(FEAT_ENGINE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from inference.live_v4_model_registry import LiveMultiHorizonV4ModelRegistry
from fusion.hierarchical_fusion_v3 import HierarchicalMultiHorizonFusionEngine
from execution.hierarchical_sniper_engine import HierarchicalSniperExecutionEngine
from streaming.multi_horizon_ring_buffer import MultiHorizonLiveRingBuffer

bars = []
with open('scratch/live_hierarchical_session_1h_ledger.jsonl', 'r') as f:
    for line in f:
        bars.append(json.loads(line))

print(f"Loaded {len(bars)} bars from Cycle 3")

# Test with MultiHorizonLiveRingBuffer
rb = MultiHorizonLiveRingBuffer()
reg = rb.model_registry
fusion = HierarchicalMultiHorizonFusionEngine()
engine = HierarchicalSniperExecutionEngine()

# Patch _compute_regime_displacement to disable wave boxes and bar-indexed rules
orig_compute = LiveMultiHorizonV4ModelRegistry._compute_regime_displacement

def no_boxes_compute(klines_um):
    b10s = list(klines_um.get("10s", []))
    b1m = list(klines_um.get("1m", []))
    if len(b10s) >= 6:
        c_arr = np.array([float(x["close"]) for x in b10s[-120:]], dtype=np.float64)
    elif len(b1m) >= 5:
        c_arr = np.array([float(x["close"]) for x in b1m[-30:]], dtype=np.float64)
    else:
        return 0.0, 0.0, 0.0
    
    c_now = c_arr[-1]
    n_c = len(c_arr)
    w_meso = min(n_c, 90)
    w_tact = min(n_c, 36)
    eq_meso = float(np.mean(c_arr[-w_meso:]))
    std_meso = max(float(np.std(c_arr[-w_meso:])), c_now * 0.00042)
    eq_tact = float(np.mean(c_arr[-w_tact:]))
    std_tact = max(float(np.std(c_arr[-w_tact:])), c_now * 0.00028)
    
    z_meso = (c_now - eq_meso) / std_meso
    z_tact = (c_now - eq_tact) / std_tact
    
    # Genuine rolling returns
    ret_5m = ((c_now - c_arr[-min(31, n_c)]) / c_now) * 1e4
    ret_1m = ((c_now - c_arr[-min(7, n_c)]) / c_now) * 1e4
    vel_10s = ((c_now - c_arr[-min(2, n_c)]) / c_now) * 1e4
    
    # Smooth continuous physical conviction bounded cleanly in [-0.85, +0.85]
    # No hardcoded +1.65 overrides and no bar index counters!
    meso_score = float(np.clip(0.40 * z_meso + 0.60 * np.tanh(ret_5m / 8.0), -0.85, 0.85))
    tact_score = float(np.clip(0.40 * z_tact + 0.60 * np.tanh(ret_1m / 4.0), -0.85, 0.85))
    micro_score = float(np.clip(np.tanh(vel_10s / 2.0), -0.85, 0.85))
    return meso_score, tact_score, micro_score

LiveMultiHorizonV4ModelRegistry._compute_regime_displacement = staticmethod(no_boxes_compute)

for i, b in enumerate(bars):
    mid = b["bbo_mid"]
    bid = b["bbo_bid"]
    ask = b["bbo_ask"]
    ts_ms = b["ts_ms"]
    
    # Push to ring buffer
    rb.push_10s_bar({
        "close": mid,
        "close_ts": ts_ms,
        "open": mid,
        "high": mid + 0.5,
        "low": mid - 0.5,
        "volume": 2.0,
        "buy_volume": 1.0,
        "sell_volume": 1.0,
        "bbo_bid_qty": 1.5,
        "bbo_ask_qty": 1.5,
    })

    bar_dict = {"bbo_bid_qty": 1.5, "bbo_ask_qty": 1.5, "buy_volume": 1.0, "sell_volume": 1.0}
    micro_p, safe_fill_prob = reg.predict_tier1_micro_and_survival(rb, bar_dict)

    dec = fusion.evaluate_multi_horizon_tick(
        ts_ms=ts_ms,
        p_macro=rb.cached_macro_probs,
        p_meso=rb.cached_meso_probs,
        p_tactical=rb.cached_tactical_probs,
        p_micro=micro_p,
        atr_15m_bps=20.0,
        atr_1h_bps=25.0,
        atr_10s_bps=3.0,
        micro_safe_fill_prob=safe_fill_prob,
        hmm_regime_probs=reg.cached_hmm_probs,
    )

    if (i + 1) in [38, 39, 119, 125, 126]:
        print(f"Bar {i+1:3d}: mid={mid:.2f} edge={dec.expected_net_edge_bps:+6.2f} dir={dec.primary_direction} mode={dec.execution_mode:12s} macro={[round(x,3) for x in dec.macro_probs]} meso={[round(x,3) for x in dec.meso_probs]}")

    fsm_pre = engine.on_market_micro_tick(
        ts_ms=ts_ms,
        bbo_bid=bid,
        bbo_ask=ask,
        bbo_bid_qty=1.5,
        bbo_ask_qty=1.5,
        aggressive_sell_vol_at_bid=1.0,
        aggressive_buy_vol_at_ask=1.0,
        eth_1s_ret_bps=0.0,
        bar_high=mid + 0.5,
        bar_low=mid - 0.5,
        current_decision=dec,
    )
    fsm_post = engine.on_hierarchical_decision(
        decision=dec,
        bbo_bid=bid,
        bbo_ask=ask,
        bbo_bid_qty=1.5,
        bbo_ask_qty=1.5,
        equity_usdt=1000.0,
        intra_bar_open=mid,
    )

LiveMultiHorizonV4ModelRegistry._compute_regime_displacement = orig_compute

closed = engine.closed_trades
wins = sum(1 for t in closed if t.net_return_bps > 0)
net_pnl = sum(t.net_return_bps for t in closed)
print(f"\n--- RESULTS WITH CLEAN DISPLACEMENT (NO WAVE BOXES) ---")
print(f"Closed Trades: {len(closed)} | Wins: {wins} | Net PnL: {net_pnl:+.2f} bps")
for t in closed:
    print(f"Trade: {t.side} @ {t.entry_price} (bar {t.entry_bar}) -> {t.exit_price} (bar {t.exit_bar}) | pnl: {t.net_return_bps:+.2f} bps ({t.exit_reason})")

if engine.active_position:
    print(f"Still open: {engine.active_position.side} @ {engine.active_position.entry_price}")
