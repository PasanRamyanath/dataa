import time
import json
import ssl
import urllib.request
from datetime import datetime

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

TUNNEL_URL = "https://charged-lying-altered-kim.trycloudflare.com/latest"
BINANCE_URL = "https://fapi.binance.com/fapi/v1/ticker/price?symbol=BTCUSDT"

def fetch_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=5, context=ctx) as r:
        return json.loads(r.read().decode("utf-8"))

print("=" * 80)
print("LIVE BTCUSDT ACCURACY & PRECISION VERIFICATION ENGINE")
print("Target Account: malaysiatrip | Dual Tesla T4s")
print("=" * 80)

history = []
start_time = time.time()
target_bars = 12  # ~2 minutes of live 10-second updates

print(f"Tracking {target_bars} live bars (10s intervals)...")

last_ts = None
while len(history) < target_bars:
    try:
        engine_res = fetch_json(TUNNEL_URL).get("latest", {})
        ts = engine_res.get("datetime")
        
        if ts != last_ts and engine_res.get("status") == "scored":
            last_ts = ts
            b_price_data = fetch_json(BINANCE_URL)
            b_price = float(b_price_data["price"])
            
            p_down = engine_res.get("prob_down", 0)
            p_flat = engine_res.get("prob_flat", 0)
            p_up = engine_res.get("prob_up", 0)
            act = engine_res.get("act_probability", 0)
            close = float(engine_res.get("close", b_price))
            direction = engine_res.get("primary_direction", 1)
            members = engine_res.get("member_predictions", {})
            
            record = {
                "bar_idx": len(history) + 1,
                "ts": ts,
                "close": close,
                "binance_price": b_price,
                "prob_down": p_down,
                "prob_flat": p_flat,
                "prob_up": p_up,
                "direction": direction,
                "act": act,
                "authorized": act >= 0.65,
                "members": members,
            }
            history.append(record)
            
            dir_str = "SHORT" if direction == 0 else ("LONG" if direction == 2 else "FLAT")
            auth_str = "[TRADE AUTHORIZED]" if record["authorized"] else "[GATED: NO CHOP]"
            print(f"Bar {record['bar_idx']:02d} | TS: {ts[-8:]} | Close: ${close:.2f} | B-Price: ${b_price:.2f} | Dir: {dir_str:<5} | Probs: [{p_down:.3f}, {p_flat:.3f}, {p_up:.3f}] | Act: {act:.3f} | {auth_str}")
        
    except Exception as e:
        print(f"Fetch tick error: {e}")
        
    time.sleep(3)

print("\n" + "=" * 80)
print("ANALYZING OUTCOMES AGAINST FORWARD PRICE REALIZATION")
print("=" * 80)

# Evaluate forward realization
fee_bps = 14.0  # Binance VIP 0 taker round-trip (5 bps + 5 bps + 4 bps slippage buffer)

for i in range(len(history) - 3):
    entry = history[i]
    fwd_1 = history[i + 1]["binance_price"]
    fwd_3 = history[i + 3]["binance_price"]
    entry_p = entry["binance_price"]
    
    ret_1 = (fwd_1 - entry_p) / entry_p * 10000.0  # in bps
    ret_3 = (fwd_3 - entry_p) / entry_p * 10000.0  # in bps
    
    dir_val = 1 if entry["direction"] == 2 else (-1 if entry["direction"] == 0 else 0)
    
    pnl_1 = (dir_val * ret_1) - (fee_bps if dir_val != 0 else 0)
    pnl_3 = (dir_val * ret_3) - (fee_bps if dir_val != 0 else 0)
    
    print(f"Bar {entry['bar_idx']:02d} | Dir: {entry['direction']} (Act: {entry['act']:.2f}) | 10s Ret: {ret_1:+.2f} bps | 30s Ret: {ret_3:+.2f} bps | Gated: {'ACTIVE' if entry['authorized'] else 'FILTERED'}")

print("\nSummary complete.")
