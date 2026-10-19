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

# Find all authorized entry bars (act >= 0.70 and direction in (0, 2))
auth_bars = [b for b in bars if b["authorized"]]
print(f"Total authorized entry bars: {len(auth_bars)}")

print("\n" + "=" * 90)
print(f"{'Entry Bar':<10} | {'Dir':<6} | {'Entry Price':<12} | {'Max Fav Excursion (MFE)':<25} | {'Bar of Peak':<12} | {'Seconds to Peak':<15}")
print("-" * 90)

for b in auth_bars:
    idx = b["bar_idx"]
    p0 = b["binance_price"]
    d = b["direction"]
    dir_mult = 1.0 if d == 2 else -1.0
    
    # Track the price over the next 30 bars (5 minutes)
    future_bars = [x for x in bars if x["bar_idx"] > idx and x["bar_idx"] <= idx + 30]
    if not future_bars:
        continue
        
    mfe_bps = -999.0
    peak_bar = idx
    for fb in future_bars:
        p_now = fb["binance_price"]
        ret_bps = dir_mult * ((p_now - p0) / p0) * 10000.0
        if ret_bps > mfe_bps:
            mfe_bps = ret_bps
            peak_bar = fb["bar_idx"]
            
    bars_to_peak = peak_bar - idx
    sec_to_peak = bars_to_peak * 10
    dir_str = "LONG" if d == 2 else "SHORT"
    print(f"Bar #{idx:<5} | {dir_str:<6} | ${p0:<10.1f} | {mfe_bps:+7.2f} bps ({mfe_bps*8.2:+5.1f} USD) | Bar +{bars_to_peak:<6} | {sec_to_peak:<4} seconds")
