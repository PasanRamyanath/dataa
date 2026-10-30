import json
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
FEAT_ENGINE_DIR = REPO_ROOT / "feature engineering scripts"
for p in [str(REPO_ROOT), str(FEAT_ENGINE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

bars = []
with open('scratch/live_hierarchical_session_1h_ledger.jsonl', 'r') as f:
    for line in f:
        bars.append(json.loads(line))

target_bars = [38, 39, 119, 125, 153, 216, 218, 257]

# Let's inspect each rule directly
klines_10s = []
for i, b in enumerate(bars):
    klines_10s.append({
        "close": b["bbo_mid"],
        "taker_imb": 0.0,
    })
    bar_num = b['bar_idx']
    if bar_num not in target_bars:
        continue

    # Run inspection
    b10s = klines_10s[-120:]
    c_arr = np.array([float(x["close"]) for x in b10s], dtype=np.float64)
    c_now = c_arr[-1]
    n_c = len(c_arr)
    w_meso = min(n_c, 90)
    w_tact = min(n_c, 36)
    eq_meso = float(np.mean(c_arr[-w_meso:]))
    std_meso = max(float(np.std(c_arr[-w_meso:])), c_now * 0.00042)
    eq_tact = float(np.mean(c_arr[-w_tact:]))
    std_tact = max(float(np.std(c_arr[-w_tact:])), c_now * 0.00028)
    std_tact_dyn = max(float(np.std(c_arr[-w_tact:])), c_now * 0.00012)
    z_meso = (c_now - eq_meso) / std_meso
    z_tact = (c_now - eq_tact) / std_tact
    z_tact_dyn = (c_now - eq_tact) / std_tact_dyn

    vel_10s = ((c_now - c_arr[-min(2, n_c)]) / c_now) * 1e4
    vel_30s = ((c_now - c_arr[-min(4, n_c)]) / c_now) * 1e4
    ret_1m = ((c_now - c_arr[-min(7, n_c)]) / c_now) * 1e4
    ret_5m = ((c_now - c_arr[-min(31, n_c)]) / c_now) * 1e4
    ret_10m = ((c_now - c_arr[-min(61, n_c)]) / c_now) * 1e4
    ret_15m = ((c_now - c_arr[-min(90, n_c)]) / c_now) * 1e4
    ret_20m = ((c_now - c_arr[0]) / c_now) * 1e4

    w_2m = min(n_c, 12)
    low_2m = float(np.min(c_arr[-w_2m:]))
    high_2m = float(np.max(c_arr[-w_2m:]))
    bnc_2m = ((c_now - low_2m) / c_now) * 1e4
    pb_2m = ((high_2m - c_now) / c_now) * 1e4

    w_6m = min(n_c, 36)
    low_6m = float(np.min(c_arr[-w_6m:]))
    high_6m = float(np.max(c_arr[-w_6m:]))
    bnc_6m = ((c_now - low_6m) / c_now) * 1e4
    dd_6m = ((high_6m - c_now) / c_now) * 1e4

    w_er = c_arr[-min(60, n_c):]
    step_sum = float(np.sum(np.abs(np.diff(w_er))))
    er_10m = float((w_er[-1] - w_er[0]) / max(step_sum, c_now * 0.0003))

    rets10_vol = np.diff(c_arr[-min(37, n_c):]) / c_arr[-min(37, n_c):-1] * 1e4 if n_c >= 3 else np.array([1.0])
    sig10 = max(float(np.std(rets10_vol)), 0.45)
    nv10 = vel_10s / sig10
    nv30 = vel_30s / sig10
    nr1 = ret_1m / sig10
    nr5 = ret_5m / sig10
    nr15 = ret_15m / sig10
    ndd6 = dd_6m / sig10
    nb6 = bnc_6m / sig10

    from inference.live_v4_model_registry import LiveMultiHorizonV4ModelRegistry

    active_rules = []
    
    r4_long = (
        n_c >= 35 and pb_2m <= 0.05 and dd_6m <= 0.60 and -0.80 <= ret_5m <= 1.50
        and 2.20 <= ret_1m <= 5.38 and vel_30s >= 1.25 and vel_10s >= 1.25
        and 3.00 <= bnc_2m <= 4.20 and vel_30s <= 3.50 and er_10m >= 0.05 and z_meso <= 0.95 and z_tact_dyn <= 1.85
    )
    if r4_long: active_rules.append("r4_long")

    r5_long = (
        n_c >= 35 and er_10m >= 0.48 and dd_6m <= 1.20 and pb_2m <= 0.15
        and vel_30s >= 0.60 and vel_10s >= 0.30 and 0.12 <= z_meso <= 2.20
        and ret_15m >= 1.50
    )
    if r5_long: active_rules.append("r5_long")

    r10_long = (
        35 <= n_c <= 55 and er_10m >= 0.68 and pb_2m <= 0.05 and dd_6m <= 0.05
        and 2.5 <= ret_5m <= 5.5 and vel_10s >= 1.45 and vel_30s >= 1.45 and bnc_2m >= 2.60
    )
    if r10_long: active_rules.append("r10_long")

    r9_long = (
        n_c >= 45 and 0.02 <= er_10m <= 0.30 and z_meso <= 1.65
        and 5.0 <= ret_15m <= 32.0 and 3.0 <= ret_10m <= 37.0
        and (dd_6m <= 13.55 or dd_6m >= 17.0)
        and vel_10s >= 0.60 and vel_30s >= 0.50 and ret_1m >= 0.50 and bnc_2m >= 0.90
    )
    if r9_long: active_rules.append("r9_long")

    klines_um = {"10s": b10s, "1m": []}
    meso, tact, micro = LiveMultiHorizonV4ModelRegistry._compute_regime_displacement(klines_um)
    print(f"Bar {bar_num:3d} (n_c={n_c:3d}): z_meso={z_meso:+.3f}, z_tact={z_tact_dyn:+.3f}, ret_1m={ret_1m:+.2f}, ret_5m={ret_5m:+.2f}, dd_6m={dd_6m:.2f}, bnc_2m={bnc_2m:.2f}, er_10m={er_10m:.2f} | active={active_rules} => meso={meso:+.2f}, tact={tact:+.2f}, micro={micro:+.2f}")
