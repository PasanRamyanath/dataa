import time
import json
import ssl
import urllib.request
import os
from datetime import datetime

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

TUNNEL_URL = "https://roughly-flip-lay-accurately.trycloudflare.com/latest"
BINANCE_PRICE_URL = "https://fapi.binance.com/fapi/v1/ticker/price?symbol=BTCUSDT"
BINANCE_BOOK_URL = "https://fapi.binance.com/fapi/v1/ticker/bookTicker?symbol=BTCUSDT"

LOG_JSONL_PATH = "scratch/monitor_30m_log.jsonl"
SUMMARY_JSON_PATH = "scratch/monitor_30m_summary.json"

TARGET_DURATION_SEC = 30 * 60  # 30 minutes = 1800 seconds (~180 bars)
POLL_INTERVAL_SEC = 2.5        # poll frequently so we catch every new bar within 2s of completion
FEE_ROUNDTRIP_BPS = 14.0       # Binance VIP 0 taker: 5 bps open + 5 bps close + 4 bps buffer

# Triple Barrier Parameters
TP_BPS = 8.0                   # +8.0 bps (+0.08%) profit take
SL_BPS = 5.0                   # -5.0 bps (-0.05%) stop loss
MAX_HOLD_BARS = 18             # 3-minute max hold horizon

def fetch_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=5, context=ctx) as r:
        return json.loads(r.read().decode("utf-8"))

print("=" * 90, flush=True)
print("STARTING 30-MINUTE REAL-TIME BTCUSDT ACCURACY & ULTRA SNIPER EVALUATION", flush=True)
print("Account: malaysiatrip | Dual Tesla T4s | Target: 180 Bars (~30 Minutes)", flush=True)
print("=" * 90, flush=True)

# Clean previous run artifacts
if os.path.exists(LOG_JSONL_PATH):
    os.remove(LOG_JSONL_PATH)

start_time = time.time()
end_time = start_time + TARGET_DURATION_SEC

bars = []
last_bar_ts = None
open_positions = []
closed_trades = []

bar_counter = 0

while time.time() < end_time:
    try:
        raw_res = fetch_json(TUNNEL_URL)
        latest = raw_res.get("latest")
        
        if latest and latest.get("status") == "scored":
            ts = latest.get("datetime")
            bar_ts = latest.get("ts")
            
            # Detect new 10s bar
            if bar_ts != last_bar_ts:
                last_bar_ts = bar_ts
                bar_counter += 1
                
                # Fetch live Binance price and order book spread
                b_price_data = fetch_json(BINANCE_PRICE_URL)
                b_book = fetch_json(BINANCE_BOOK_URL)
                
                b_price = float(b_price_data["price"])
                bid = float(b_book["bidPrice"])
                ask = float(b_book["askPrice"])
                spread_bps = ((ask - bid) / b_price) * 10000.0
                
                p_down = float(latest.get("prob_down", 0.0))
                p_flat = float(latest.get("prob_flat", 0.0))
                p_up = float(latest.get("prob_up", 0.0))
                act = float(latest.get("act_probability", 0.0))
                direction = int(latest.get("primary_direction", 1))
                close = float(latest.get("close", b_price))
                members = latest.get("member_predictions", {})
                
                is_authorized = act >= 0.70 and direction in (0, 2)
                dir_name = "SHORT" if direction == 0 else ("LONG" if direction == 2 else "FLAT")
                auth_tag = "[ULTRA SNIPER ACTIVE]" if is_authorized else "[CHOP VETOED]"
                
                bar_record = {
                    "bar_idx": bar_counter,
                    "ts": ts,
                    "bar_ts": bar_ts,
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
                bars.append(bar_record)
                
                # Append to JSONL log
                with open(LOG_JSONL_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps(bar_record) + "\n")
                
                # Check / update existing open triple-barrier positions
                active_pos = []
                for pos in open_positions:
                    hold_bars = bar_counter - pos["entry_bar"]
                    dir_mult = 1.0 if pos["direction"] == 2 else -1.0
                    unrealized_bps = dir_mult * ((b_price - pos["entry_price"]) / pos["entry_price"]) * 10000.0
                    
                    hit_tp = unrealized_bps >= TP_BPS
                    hit_sl = unrealized_bps <= -SL_BPS
                    expired = hold_bars >= MAX_HOLD_BARS
                    
                    if hit_tp or hit_sl or expired:
                        exit_reason = "TAKE_PROFIT" if hit_tp else ("STOP_LOSS" if hit_sl else "TIME_EXPIRY")
                        net_pnl_bps = unrealized_bps - FEE_ROUNDTRIP_BPS
                        pnl_100x_margin_pct = net_pnl_bps * 1.0  # at 100x leverage, 1 bps = 1% on margin
                        
                        closed_trade = {
                            "trade_id": len(closed_trades) + 1,
                            "entry_bar": pos["entry_bar"],
                            "exit_bar": bar_counter,
                            "hold_bars": hold_bars,
                            "direction": pos["direction"],
                            "dir_name": "LONG" if pos["direction"] == 2 else "SHORT",
                            "entry_price": pos["entry_price"],
                            "exit_price": b_price,
                            "exit_reason": exit_reason,
                            "gross_bps": unrealized_bps,
                            "fee_bps": FEE_ROUNDTRIP_BPS,
                            "net_bps": net_pnl_bps,
                            "pnl_100x_pct": pnl_100x_margin_pct,
                            "is_win": net_pnl_bps > 0,
                        }
                        closed_trades.append(closed_trade)
                        print(f"  >>> TRADE #{closed_trade['trade_id']} CLOSED [{exit_reason}]: {closed_trade['dir_name']} | "
                              f"Entry: ${pos['entry_price']:.2f} -> Exit: ${b_price:.2f} | "
                              f"Gross: {unrealized_bps:+.2f} bps | Net: {net_pnl_bps:+.2f} bps | 100x: {pnl_100x_margin_pct:+.1f}%", flush=True)
                    else:
                        active_pos.append(pos)
                open_positions = active_pos
                
                # If authorized, open a new position
                if is_authorized:
                    open_positions.append({
                        "entry_bar": bar_counter,
                        "direction": direction,
                        "entry_price": b_price,
                        "act": act,
                    })
                
                # Compute live stats
                n_vetoed = sum(1 for b in bars if not b["authorized"])
                fee_saved = n_vetoed * FEE_ROUNDTRIP_BPS
                
                n_closed = len(closed_trades)
                wins = sum(1 for t in closed_trades if t["is_win"])
                win_rate = (wins / n_closed * 100.0) if n_closed > 0 else 0.0
                total_net_bps = sum(t["net_bps"] for t in closed_trades)
                avg_net_bps = (total_net_bps / n_closed) if n_closed > 0 else 0.0
                
                gross_wins = sum(t["gross_bps"] for t in closed_trades if t["gross_bps"] > 0)
                gross_losses = abs(sum(t["gross_bps"] for t in closed_trades if t["gross_bps"] < 0))
                profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else (99.0 if gross_wins > 0 else 1.0)
                
                # Print live bar summary
                elapsed_min = (time.time() - start_time) / 60.0
                print(f"Bar {bar_counter:03d} ({elapsed_min:.1f}m) | TS: {ts[-8:]} | Price: ${b_price:.2f} (Spread: {spread_bps:.1f} bps) | "
                      f"Dir: {dir_name:<5} | P: [{p_down:.3f}, {p_flat:.3f}, {p_up:.3f}] | Act: {act:.3f} | {auth_tag}", flush=True)
                
                # Save rolling summary
                summary = {
                    "elapsed_minutes": round(elapsed_min, 2),
                    "total_bars": bar_counter,
                    "target_bars": 180,
                    "chop_vetoed_bars": n_vetoed,
                    "chop_veto_pct": round((n_vetoed / bar_counter) * 100.0, 1),
                    "fees_saved_bps": round(fee_saved, 1),
                    "authorized_trades_closed": n_closed,
                    "authorized_open_positions": len(open_positions),
                    "wins": wins,
                    "losses": n_closed - wins,
                    "precision_win_rate_pct": round(win_rate, 1),
                    "total_net_bps": round(total_net_bps, 2),
                    "average_net_bps_per_trade": round(avg_net_bps, 2),
                    "profit_factor": round(profit_factor, 2),
                    "latest_price": b_price,
                    "updated_at": datetime.utcnow().isoformat(),
                }
                with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as sf:
                    json.dump(summary, sf, indent=2)
                    
    except Exception as e:
        print(f"Live polling loop warning: {e}", flush=True)
        
    time.sleep(POLL_INTERVAL_SEC)

print("\n" + "=" * 90, flush=True)
print("30-MINUTE MONITORING COMPLETE! FINAL AUDIT SUMMARY:", flush=True)
print("=" * 90, flush=True)
if os.path.exists(SUMMARY_JSON_PATH):
    with open(SUMMARY_JSON_PATH, "r") as sf:
        final_summary = json.load(sf)
    print(json.dumps(final_summary, indent=2), flush=True)
