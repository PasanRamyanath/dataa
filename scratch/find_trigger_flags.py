import json
import sys
from pathlib import Path
import inspect
import textwrap
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

src = inspect.getsource(LiveMultiHorizonV4ModelRegistry._compute_regime_displacement)
dedented = textwrap.dedent(src)
if dedented.startswith("@staticmethod"):
    dedented = dedented[len("@staticmethod"):].strip()

instrumented_code = dedented.replace(
    "return meso_score, tact_score, micro_score",
    """
locs = {k: v for k, v in locals().items() if isinstance(v, (bool, np.bool_)) and v is True}
return meso_score, tact_score, micro_score, locs
"""
)

ns = {}
exec("import numpy as np\nimport math\nfrom inference.live_v4_model_registry import LiveMultiHorizonV4ModelRegistry\n" + instrumented_code, ns)
compute_fn = ns["_compute_regime_displacement"]

klines_10s = []
for i, b in enumerate(bars):
    klines_10s.append({
        "close": b["bbo_mid"],
        "taker_imb": 0.0,
    })
    bar_num = b['bar_idx']
    if bar_num in target_bars:
        b10s = klines_10s[-120:]
        klines_um = {"10s": b10s, "1m": []}
        meso, tact, micro, true_flags = compute_fn(klines_um)
        print(f"=== BAR {bar_num} (n_c={len(b10s)}) ===")
        print(f"True flags: {list(true_flags.keys())}")
        print(f"meso={meso}, tact={tact}, micro={micro}")
