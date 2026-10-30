import json
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
FEAT_ENGINE_DIR = REPO_ROOT / "feature engineering scripts"
for p in [str(REPO_ROOT), str(FEAT_ENGINE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from inference.live_v4_model_registry import LiveMultiHorizonV4ModelRegistry, _score_to_simplex
from fusion.hierarchical_fusion_v3 import HierarchicalMultiHorizonFusionEngine
from execution.hierarchical_sniper_engine import HierarchicalSniperExecutionEngine

# Load bars from Cycle 3
bars = []
with open('scratch/live_hierarchical_session_1h_ledger.jsonl', 'r') as f:
    for line in f:
        bars.append(json.loads(line))

print(f"Loaded {len(bars)} bars from Cycle 3")

# Define clean displacement
def clean_displacement(klines_um):
    b10s = list(klines_um.get("10s", []))
    if len(b10s) < 6:
        return 0.0, 0.0, 0.0
    c_arr = np.array([float(x["close"]) for x in b10s[-60:]], dtype=np.float64)
    c_now = c_arr[-1]
    
    # 5m return & 1m return
    ret_5m = (c_now - c_arr[-min(30, len(c_arr))]) / c_now * 1e4
    ret_1m = (c_now - c_arr[-min(6, len(c_arr))]) / c_now * 1e4
    vel_10s = (c_now - c_arr[-2]) / c_now * 1e4
    
    vol = max(float(np.std(np.diff(c_arr) / c_arr[:-1] * 1e4)), 0.8)
    
    # Continuous normalized Z-scores bounded smoothly in [-1.2, +1.2]
    meso = float(np.clip(ret_5m / (vol * 3.5), -1.2, 1.2))
    tact = float(np.clip(ret_1m / (vol * 2.0), -1.2, 1.2))
    micro = float(np.clip(vel_10s / (vol * 1.2), -1.2, 1.2))
    return meso, tact, micro

# Patch displacement
orig_compute = LiveMultiHorizonV4ModelRegistry._compute_regime_displacement
LiveMultiHorizonV4ModelRegistry._compute_regime_displacement = staticmethod(clean_displacement)

# Let's inspect the signals at Bar 38, 39, 119, 125 with clean displacement
reg = LiveMultiHorizonV4ModelRegistry()
fusion = HierarchicalMultiHorizonFusionEngine()
engine = HierarchicalSniperExecutionEngine()

class MockRingBuffer:
    def __init__(self):
        self.klines_um = {"10s": [], "1m": [], "5m": [], "15m": [], "1h": [], "4h": []}

rb = MockRingBuffer()
signals = []

for i, b in enumerate(bars):
    mid = b["bbo_mid"]
    bid = b["bbo_bid"]
    ask = b["bbo_ask"]
    ts_ms = b["ts_ms"]
    
    rb.klines_um["10s"].append({"close": mid, "taker_imb": 0.0})
    if len(rb.klines_um["10s"]) > 120:
        rb.klines_um["10s"].pop(0)

    # Inferences
    macro_p = reg.predict_tier4_macro(rb)
    meso_p = reg.predict_tier3_meso(rb)
    tact_p = reg.predict_tier2_tactical(rb)
    bar_dict = {"bbo_bid_qty": 1.5, "bbo_ask_qty": 1.5, "buy_volume": 1.0, "sell_volume": 1.0}
    micro_p, safe_fill_prob = reg.predict_tier1_micro_and_survival(rb, bar_dict)

    # Fusion
    dec = fusion.evaluate_multi_horizon_tick(
        ts_ms=ts_ms,
        p_macro=macro_p,
        p_meso=meso_p,
        p_tactical=tact_p,
        p_micro=micro_p,
        atr_15m_bps=20.0,
        atr_1h_bps=25.0,
        atr_10s_bps=3.0,
        micro_safe_fill_prob=safe_fill_prob,
        hmm_regime_probs=reg.cached_hmm_probs,
    )

    if (i + 1) in [38, 39, 119, 125]:
        print(f"Bar {i+1:3d}: edge={dec.expected_net_edge_bps:+6.2f} dir={dec.primary_direction} mode={dec.execution_mode:12s} macro={[round(x,3) for x in dec.macro_probs]} meso={[round(x,3) for x in dec.meso_probs]}")

    # Advance engine
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

print(f"\nTotal closed trades: {len(engine.closed_trades)}")
for t in engine.closed_trades:
    print(f"Trade: {t.side} @ {t.entry_price} -> {t.exit_price} | pnl: {t.net_return_bps:+.2f} bps ({t.exit_reason})")

if engine.active_position:
    print(f"Still open: {engine.active_position.side} @ {engine.active_position.entry_price}")
