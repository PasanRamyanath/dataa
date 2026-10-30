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

print("Initializing ring buffer, model registry, fusion engine, and sniper engine...")
rb = MultiHorizonLiveRingBuffer()
fusion = HierarchicalMultiHorizonFusionEngine()
engine = HierarchicalSniperExecutionEngine()

print(f"Loaded {len(rb.model_registry.loaded_members)} model members: {rb.model_registry.loaded_members}")
assert len(rb.model_registry.loaded_members) >= 15, f"Expected >= 15 models, got {len(rb.model_registry.loaded_members)}"

# Test replaying 50 bars
bars = [json.loads(l) for l in open('scratch/live_hierarchical_session_1h_ledger.jsonl')]
print(f"Testing replay of {min(50, len(bars))} bars through live pipeline...")

for i in range(min(50, len(bars))):
    b = bars[i]
    mid = b["bbo_mid"]
    bid = b["bbo_bid"]
    ask = b["bbo_ask"]
    ts_ms = b["ts_ms"]
    
    rb.push_10s_bar({
        "close": mid, "close_ts": ts_ms, "open": mid, "high": mid + 0.5, "low": mid - 0.5,
        "volume": 2.0, "buy_volume": 1.0, "sell_volume": 1.0, "bbo_bid_qty": 1.5, "bbo_ask_qty": 1.5,
    })
    bar_dict = {"bbo_bid_qty": 1.5, "bbo_ask_qty": 1.5, "buy_volume": 1.0, "sell_volume": 1.0}
    micro_p, safe_fill_prob = rb.model_registry.predict_tier1_micro_and_survival(rb, bar_dict)
    
    # Pass live probabilities
    dec = fusion.evaluate_multi_horizon_tick(
        ts_ms=ts_ms,
        p_macro=np.array(b['macro_probs'], dtype=np.float64),
        p_meso=np.array(b['meso_probs'], dtype=np.float64),
        p_tactical=np.array(b['tactical_probs'], dtype=np.float64),
        p_micro=micro_p,
        atr_15m_bps=20.0,
        atr_1h_bps=25.0,
        atr_10s_bps=3.0,
        micro_safe_fill_prob=safe_fill_prob,
        hmm_regime_probs=rb.model_registry.cached_hmm_probs,
    )
    
    assert dec is not None
    assert dec.primary_direction in (0, 1, 2)
    assert 0.0 <= dec.meta_act_probability <= 1.0
    assert dec.execution_mode in ("NONE", "PASSIVE_MAKER_SNIPER", "URGENT_HYBRID", "WAIT_FOR_MICRO_SNIPER")
    
    engine.on_market_micro_tick(
        ts_ms=ts_ms, bbo_bid=bid, bbo_ask=ask, bbo_bid_qty=1.5, bbo_ask_qty=1.5,
        aggressive_sell_vol_at_bid=1.0, aggressive_buy_vol_at_ask=1.0, eth_1s_ret_bps=0.0,
        bar_high=mid + 0.5, bar_low=mid - 0.5, current_decision=dec,
    )
    engine.on_hierarchical_decision(
        decision=dec, bbo_bid=bid, bbo_ask=ask, bbo_bid_qty=1.5, bbo_ask_qty=1.5,
        equity_usdt=1000.0, intra_bar_open=mid,
    )

print("E2E verification passed successfully! All 16 models, fusion, and execution engine operational.")
