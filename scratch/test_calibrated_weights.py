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

bars = []
with open('scratch/live_hierarchical_session_1h_ledger.jsonl', 'r') as f:
    for line in f:
        bars.append(json.loads(line))

print(f"Loaded {len(bars)} bars from Cycle 3")

# Let's inspect neural trajectory step_slope across bars
# We can create a ring buffer with the actual data from Cycle 3
from streaming.multi_horizon_ring_buffer import MultiHorizonLiveRingBuffer

rb = MultiHorizonLiveRingBuffer()
# Bootstrap or load 10s bars
print("Feeding bars into ring buffer...")
for i, b in enumerate(bars):
    rb.push_10s_bar({
        "close": b["bbo_mid"],
        "close_ts": b["ts_ms"],
        "open": b["bbo_mid"],
        "high": b["bbo_mid"] + 0.5,
        "low": b["bbo_mid"] - 0.5,
        "volume": 2.0,
        "buy_volume": 1.0,
        "sell_volume": 1.0,
        "bbo_bid_qty": 1.5,
        "bbo_ask_qty": 1.5,
    })

print(f"Ring buffer populated: 10s={len(rb.klines_um['10s'])}, 1m={len(rb.klines_um['1m'])}, 5m={len(rb.klines_um['5m'])}, 15m={len(rb.klines_um['15m'])}")

# Check predictions with current weights
print("\n--- Current Tier Predictions ---")
print("Macro probs:   ", [round(x, 4) for x in rb.cached_macro_probs])
print("Meso probs:    ", [round(x, 4) for x in rb.cached_meso_probs])
print("Tactical probs:", [round(x, 4) for x in rb.cached_tactical_probs])
