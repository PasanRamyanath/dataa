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

bars = []
with open('scratch/live_hierarchical_session_1h_ledger.jsonl', 'r') as f:
    for line in f:
        bars.append(json.loads(line))

target_bars = [38, 39, 119, 125]
captured_locals = {}

def tracer(frame, event, arg):
    if event == 'return' and frame.f_code.co_name == '_compute_regime_displacement':
        locs = frame.f_locals
        true_keys = [k for k, v in locs.items() if isinstance(v, (bool, np.bool_)) and v is True]
        captured_locals[current_bar] = true_keys
    return tracer

klines_10s = []
for i, b in enumerate(bars):
    klines_10s.append({
        "close": b["bbo_mid"],
        "taker_imb": 0.0,
    })
    bar_num = b['bar_idx']
    if bar_num in target_bars:
        current_bar = bar_num
        b10s = klines_10s[-120:]
        klines_um = {"10s": b10s, "1m": []}
        sys.settrace(tracer)
        meso, tact, micro = LiveMultiHorizonV4ModelRegistry._compute_regime_displacement(klines_um)
        sys.settrace(None)
        print(f"=== BAR {bar_num} (n_c={len(b10s)}) ===")
        print(f"True boolean flags: {captured_locals.get(bar_num, [])}")
        print(f"meso={meso:.2f}, tact={tact:.2f}, micro={micro:.2f}")
