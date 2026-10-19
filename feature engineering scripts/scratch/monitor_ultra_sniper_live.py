import time
import json
import ssl
import urllib.request
from datetime import datetime

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

TUNNEL_URL = "https://roughly-flip-lay-accurately.trycloudflare.com/latest"
BINANCE_PRICE_URL = "https://fapi.binance.com/fapi/v1/ticker/price?symbol=BTCUSDT"
BINANCE_BOOK_URL = "https://fapi.binance.com/fapi/v1/ticker/bookTicker?symbol=BTCUSDT"

def fetch_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=5, context=ctx) as r:
        return json.loads(r.read().decode("utf-8"))

print("=" * 85)
print("MONITORING ULTRA SNIPER ENGINE ON REAL LIVE BINANCE USD(S)-M FUTURES DATA")
print("Threshold tau = 0.70 | Target Account: malaysiatrip | Dual Tesla T4s")
print("=" * 85)

history = []
target_bars = 15  # ~2.5 minutes of 10s bars
last_ts = None

while len(history) < target_bars:
    try:
        raw_res = fetch_json(TUNNEL_URL)
        latest = raw_res.get("latest")
        if latest and latest.get("status") == "scored":
            ts = latest.get("datetime")
            if ts != last_ts:
                last_ts = ts
                b_price = float(fetch_json(BINANCE_PRICE_URL)["price"])
                b_book = fetch_json(BINANCE_BOOK_URL)
                bid = float(b_book["bidPrice"])
                ask = float(b_book["askPrice"])
                spread_bps = ((ask - bid) / b_price) * 10000.0

                p_down = latest.get("prob_down", 0)
                p_flat = latest.get("prob_flat", 0)
                p_up = latest.get("prob_up", 0)
                act = latest.get("act_probability", 0)
                direction = latest.get("primary_direction", 1)
                close = float(latest.get("close", b_price))
                members = latest.get("member_predictions", {})

                is_authorized = act >= 0.70
                dir_name = "SHORT" if direction == 0 else ("LONG" if direction == 2 else "FLAT")
                auth_tag = "[ULTRA SNIPER ACTIVE]" if is_authorized else "[CHOP VETOED]"

                entry = {
                    "bar": len(history) + 1,
                    "ts": ts,
                    "close": close,
                    "binance_price": b_price,
                    "bid": bid,
                    "ask": ask,
                    "spread_bps": spread_bps,
                    "p_down": p_down,
                    "p_flat": p_flat,
                    "p_up": p_up,
                    "direction": direction,
                    "act": act,
                    "authorized": is_authorized,
                    "members": members,
                }
                history.append(entry)

                print(
                    f"Bar {entry['bar']:02d} | TS: {ts[-8:]} | Price: ${b_price:.2f} (Spread: {spread_bps:.1f} bps) "
                    f"| Dir: {dir_name:<5} | P: [{p_down:.3f}, {p_flat:.3f}, {p_up:.3f}] | Act: {act:.3f} | {auth_tag}",
                    flush=True
                )
    except Exception as e:
        print(f"Polling warning: {e}", flush=True)

    time.sleep(3)

print("\n" + "=" * 85)
print("REAL-TIME FORWARD EVALUATION AGAINST LIVE TICK DISPLACEMENTS")
print("=" * 85)

fee_roundtrip_bps = 14.0  # Binance VIP 0 taker: 5 bps open + 5 bps close + 4 bps slippage buffer

authorized_trades = []
vetoed_bars = []

for i in range(len(history) - 3):
    h = history[i]
    fwd_10s = history[i + 1]["binance_price"]
    fwd_30s = history[i + 3]["binance_price"]
    entry_p = h["binance_price"]

    ret_10s_bps = ((fwd_10s - entry_p) / entry_p) * 10000.0
    ret_30s_bps = ((fwd_30s - entry_p) / entry_p) * 10000.0

    dir_mult = 1.0 if h["direction"] == 2 else (-1.0 if h["direction"] == 0 else 0.0)
    gross_pnl_30s = dir_mult * ret_30s_bps
    net_pnl_30s = gross_pnl_30s - (fee_roundtrip_bps if dir_mult != 0 else 0.0)

    is_correct = (dir_mult > 0 and ret_30s_bps > 0) or (dir_mult < 0 and ret_30s_bps < 0) or (dir_mult == 0 and abs(ret_30s_bps) < 1.5)

    res_dict = {
        "bar": h["bar"],
        "dir": h["direction"],
        "act": h["act"],
        "authorized": h["authorized"],
        "ret_10s_bps": ret_10s_bps,
        "ret_30s_bps": ret_30s_bps,
        "net_pnl_30s": net_pnl_30s,
        "correct": is_correct,
    }

    if h["authorized"]:
        authorized_trades.append(res_dict)
    else:
        vetoed_bars.append(res_dict)

    status_lbl = "WIN" if (h["authorized"] and is_correct) else ("LOSS" if h["authorized"] else "CHOP_SKIPPED")
    print(
        f"Bar {h['bar']:02d} | Dir: {h['direction']} (Act: {h['act']:.3f}) | 10s: {ret_10s_bps:+.2f} bps | 30s: {ret_30s_bps:+.2f} bps "
        f"| Net PnL (after fees): {net_pnl_30s:+.2f} bps | Result: {status_lbl}",
        flush=True
    )

print("\n" + "-" * 85)
print(f"TOTAL EVALUATED: {len(history) - 3} bars")
if authorized_trades:
    auth_wins = sum(1 for t in authorized_trades if t["correct"])
    auth_wr = (auth_wins / len(authorized_trades)) * 100.0
    avg_net = sum(t["net_pnl_30s"] for t in authorized_trades) / len(authorized_trades)
    print(f"[SNIPER] ULTRA SNIPER AUTHORIZED TRADES: {len(authorized_trades)} | Wins: {auth_wins} | Precision Win Rate: {auth_wr:.1f}%")
    print(f"[SNIPER] AVERAGE NET PROFIT PER TRADE (after 14 bps VIP 0 fees): {avg_net:+.2f} bps")
else:
    print("[SNIPER] ULTRA SNIPER AUTHORIZED TRADES: 0 (Strict preservation of capital during low-conviction consolidation)")

print(f"[SHIELD] CHOP VETOED BARS: {len(vetoed_bars)} (Saved ~{len(vetoed_bars) * fee_roundtrip_bps:.1f} bps in churn fees)")
print("=" * 85)
