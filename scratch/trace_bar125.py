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

klines_10s = []
for i, b in enumerate(bars):
    klines_10s.append({
        "close": b["bbo_mid"],
        "taker_imb": 0.0,
    })
    bar_num = b['bar_idx']
    if bar_num == 125:
        b10s = klines_10s[-120:]
        klines_um = {"10s": b10s, "1m": []}
        
        # Trace line by line
        def tracer(frame, event, arg):
            if frame.f_code.co_name == '_compute_regime_displacement':
                lineno = frame.f_lineno
                locs = frame.f_locals
                if 'is_long_any' in locs and locs['is_long_any']:
                    for k in ['is_long_any', 'r28_dimless_long', 'live_adaptive_long', 'ms_phys_long',
                              'p1_long', 'p2_long', 'p3_long', 'p6_long', 'p7_long',
                              'r2_long', 'r4_long', 'r5_long', 'r7_long', 'r9_long',
                              'r10_long', 'r11_long', 'r12_long', 'r13_long', 'r14_long',
                              'r15_long', 'r16_long', 'r17_long', 'r18_long', 'r19_long',
                              'r20_long', 'r21_long', 'r22_long', 'r23_long', 'r24_long',
                              'r25_long', 'r26_long']:
                        if locs.get(k):
                            print(f"  --> {k} = {locs.get(k)}")
            return tracer

        sys.settrace(tracer)
        meso, tact, micro = LiveMultiHorizonV4ModelRegistry._compute_regime_displacement(klines_um)
        sys.settrace(None)
        break
