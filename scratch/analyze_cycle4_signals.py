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
from streaming.multi_horizon_ring_buffer import MultiHorizonLiveRingBuffer

bars = [json.loads(l) for l in open('scratch/live_hierarchical_session_1h_ledger.jsonl')]
rb = MultiHorizonLiveRingBuffer()
reg = rb.model_registry

macro_edges = []
meso_edges = []
tact_edges = []
micro_edges = []

for i, b in enumerate(bars):
    mid = b["bbo_mid"]
    ts_ms = b["ts_ms"]
    rb.push_10s_bar({
        "close": mid, "close_ts": ts_ms, "open": mid, "high": mid + 0.5, "low": mid - 0.5,
        "volume": 2.0, "buy_volume": 1.0, "sell_volume": 1.0, "bbo_bid_qty": 1.5, "bbo_ask_qty": 1.5,
    })
    bar_dict = {"bbo_bid_qty": 1.5, "bbo_ask_qty": 1.5, "buy_volume": 1.0, "sell_volume": 1.0}
    micro_p, _ = reg.predict_tier1_micro_and_survival(rb, bar_dict)
    
    p_mac = rb.cached_macro_probs
    p_mes = rb.cached_meso_probs
    p_tac = rb.cached_tactical_probs
    
    macro_edges.append(float(p_mac[2] - p_mac[0]))
    meso_edges.append(float(p_mes[2] - p_mes[0]))
    tact_edges.append(float(p_tac[2] - p_tac[0]))
    micro_edges.append(float(micro_p[2] - micro_p[0]))

print(f"Total bars: {len(bars)}")
print(f"Macro edge (p_up - p_dn): mean={np.mean(macro_edges):.4f}, min={np.min(macro_edges):.4f}, max={np.max(macro_edges):.4f}")
print(f"Meso edge  (p_up - p_dn): mean={np.mean(meso_edges):.4f}, min={np.min(meso_edges):.4f}, max={np.max(meso_edges):.4f}")
print(f"Tact edge  (p_up - p_dn): mean={np.mean(tact_edges):.4f}, min={np.min(tact_edges):.4f}, max={np.max(tact_edges):.4f}")
print(f"Micro edge (p_up - p_dn): mean={np.mean(micro_edges):.4f}, min={np.min(micro_edges):.4f}, max={np.max(micro_edges):.4f}")

# Look at distribution of signs
print(f"Macro: positive={sum(x > 0.05 for x in macro_edges)}, negative={sum(x < -0.05 for x in macro_edges)}, flat={sum(abs(x) <= 0.05 for x in macro_edges)}")
print(f"Meso:  positive={sum(x > 0.05 for x in meso_edges)}, negative={sum(x < -0.05 for x in meso_edges)}, flat={sum(abs(x) <= 0.05 for x in meso_edges)}")
print(f"Tact:  positive={sum(x > 0.05 for x in tact_edges)}, negative={sum(x < -0.05 for x in tact_edges)}, flat={sum(abs(x) <= 0.05 for x in tact_edges)}")
print(f"Micro: positive={sum(x > 0.05 for x in micro_edges)}, negative={sum(x < -0.05 for x in micro_edges)}, flat={sum(abs(x) <= 0.05 for x in micro_edges)}")
