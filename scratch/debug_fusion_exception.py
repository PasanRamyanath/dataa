import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
FEAT_ENGINE_DIR = REPO_ROOT / "feature engineering scripts"
for p in [str(REPO_ROOT), str(FEAT_ENGINE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import json
from streaming.multi_horizon_ring_buffer import MultiHorizonLiveRingBuffer
from fusion.hierarchical_fusion_v3 import HierarchicalMultiHorizonFusionEngine

bars = []
with open('scratch/live_hierarchical_session_1h_ledger.jsonl', 'r') as f:
    for line in f:
        bars.append(json.loads(line))

rb = MultiHorizonLiveRingBuffer()
reg = rb.model_registry
fusion = HierarchicalMultiHorizonFusionEngine()

# Lower thresholds to see where it enters
fusion.p_fl_threshold = 0.55
fusion.edge_threshold = 0.15

for i, b in enumerate(bars):
    mid = b["bbo_mid"]
    ts_ms = b["ts_ms"]
    rb.push_10s_bar({
        "close": mid, "close_ts": ts_ms, "open": mid, "high": mid + 0.5, "low": mid - 0.5,
        "volume": 2.0, "buy_volume": 1.0, "sell_volume": 1.0, "bbo_bid_qty": 1.5, "bbo_ask_qty": 1.5,
    })
    bar_dict = {"bbo_bid_qty": 1.5, "bbo_ask_qty": 1.5, "buy_volume": 1.0, "sell_volume": 1.0}
    micro_p, safe_fill_prob = reg.predict_tier1_micro_and_survival(rb, bar_dict)

    try:
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
    except Exception as e:
        print(f"Exception at bar {i+1}: {e}")
        import traceback
        traceback.print_exc()
        break
