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

# Load bars from Cycle 3
bars = []
with open('scratch/live_hierarchical_session_1h_ledger.jsonl', 'r') as f:
    for line in f:
        bars.append(json.loads(line))

print(f"Loaded {len(bars)} bars from Cycle 3")

def run_simulation(displacement_mode="CURRENT"):
    reg = LiveMultiHorizonV4ModelRegistry()
    fusion = HierarchicalMultiHorizonFusionEngine()
    engine = HierarchicalSniperExecutionEngine()

    # Patch displacement if requested
    orig_compute = LiveMultiHorizonV4ModelRegistry._compute_regime_displacement

    if displacement_mode == "SMOOTH_PHYSICS":
        def smooth_compute(klines_um):
            b10s = list(klines_um.get("10s", []))
            if len(b10s) < 6:
                return 0.0, 0.0, 0.0
            c_arr = np.array([float(x["close"]) for x in b10s[-60:]], dtype=np.float64)
            c_now = c_arr[-1]
            # Smooth rolling return & momentum
            ret_1m = (c_now - c_arr[-min(6, len(c_arr))]) / c_now * 1e4
            ret_5m = (c_now - c_arr[-min(30, len(c_arr))]) / c_now * 1e4
            std_ret = max(float(np.std(np.diff(c_arr) / c_arr[:-1] * 1e4)), 1.0)
            
            # Continuous normalized z-scores (bounded to -1.5, +1.5)
            meso = float(np.clip(ret_5m / (std_ret * 3.0), -1.5, 1.5))
            tact = float(np.clip(ret_1m / (std_ret * 1.5), -1.5, 1.5))
            micro = float(np.clip((c_now - c_arr[-2]) / c_now * 1e4 / std_ret, -1.5, 1.5))
            return meso, tact, micro

        LiveMultiHorizonV4ModelRegistry._compute_regime_displacement = staticmethod(smooth_compute)

    elif displacement_mode == "PURE_ML":
        LiveMultiHorizonV4ModelRegistry._compute_regime_displacement = staticmethod(lambda k: (0.0, 0.0, 0.0))

    trades_history = []
    signals = []

    # Mock ring buffer
    class MockRingBuffer:
        def __init__(self):
            self.klines_um = {"10s": [], "1m": [], "15m": [], "1h": [], "4h": []}

    rb = MockRingBuffer()

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
        fused = fusion.fuse_horizons(macro_p, meso_p, tact_p, micro_p, safe_fill_prob)

        # Sniper evaluation
        sig = engine.on_bar_telemetry(
            bar_idx=i + 1,
            ts_ms=ts_ms,
            bbo_bid=bid,
            bbo_ask=ask,
            macro_probs=macro_p,
            meso_probs=meso_p,
            tactical_probs=tact_p,
            micro_probs=micro_p,
            safe_fill_prob=safe_fill_prob,
            fused_decision=fused,
        )
        signals.append((i + 1, sig.execution_mode, sig.primary_direction, fused.expected_net_edge_bps))

    # Restore
    LiveMultiHorizonV4ModelRegistry._compute_regime_displacement = orig_compute

    closed = engine.closed_trades
    wins = sum(1 for t in closed if t.net_return_bps > 0)
    net_pnl = sum(t.net_return_bps for t in closed)
    print(f"\n--- MODE: {displacement_mode} ---")
    print(f"Closed Trades: {len(closed)} | Wins: {wins} | Net PnL: {net_pnl:+.2f} bps")
    for t in closed:
        print(f"  Trade: {t.side} @ {t.entry_price:.2f} (bar {t.entry_bar}) -> {t.exit_price:.2f} (bar {t.exit_bar}) | pnl: {t.net_return_bps:+.2f} bps ({t.exit_reason})")
    
    # Active position at end?
    if engine.active_position:
        ap = engine.active_position
        print(f"  Still Open: {ap.side} @ {ap.entry_price:.2f} (bar {ap.entry_bar})")

run_simulation("CURRENT")
run_simulation("SMOOTH_PHYSICS")
run_simulation("PURE_ML")
