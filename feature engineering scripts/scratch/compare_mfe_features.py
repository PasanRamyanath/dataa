import json
import numpy as np

LOG_PATH = "feature engineering scripts/scratch/monitor_30m_log.jsonl"

def load_bars():
    bars = []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                bars.append(json.loads(line.strip()))
    return bars

bars = load_bars()
auth_bars = [b for b in bars if b["authorized"]]

for b in auth_bars:
    idx = b["bar_idx"]
    p0 = b["binance_price"]
    d = b["direction"]
    dir_mult = 1.0 if d == 2 else -1.0
    future = [x for x in bars if x["bar_idx"] > idx and x["bar_idx"] <= idx + 30]
    mfe_bps = max([dir_mult * ((fb["binance_price"] - p0) / p0) * 10000.0 for fb in future]) if future else 0.0
    b["mfe_bps"] = mfe_bps

print(f"{'Bar':<5} | {'MFE (bps)':<10} | {'Act Proba':<10} | {'P(Down)':<10} | {'P(Flat)':<10} | {'Spread (bps)':<12} | {'GBM':<8} | {'iTrans':<8} | {'TFT':<8} | {'PatchTST':<8}")
print("-" * 95)
for b in auth_bars:
    m = b["members"]
    gbm_d = m["gbm-family"][0]
    it_d = m["itransformer"][0]
    tft_d = m["tft"][0]
    pt_d = m["patchtst"][0]
    print(f"#{b['bar_idx']:<4} | {b['mfe_bps']:+7.2f}    | {b['act']:8.4f}   | {b['p_down']:8.4f}   | {b['p_flat']:8.4f}   | {b['spread_bps']:10.4f}   | {gbm_d:6.3f}   | {it_d:6.3f}   | {tft_d:6.3f}   | {pt_d:6.3f}")
