import json
import time
import urllib.request
from datetime import datetime, timezone

def main():
    print("Connecting to live Kaggle VM tunnel and Binance Futures live feed...")
    tunnel_url = "https://analyses-organized-enables-mechanism.trycloudflare.com/latest"
    binance_url = "https://fapi.binance.com/fapi/v1/ticker/price?symbol=BTCUSDT"

    samples = []
    seen_ts = set()

    for i in range(12):
        try:
            req = urllib.request.Request(tunnel_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode()).get("latest", {})

            b_req = urllib.request.Request(binance_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(b_req, timeout=5) as resp:
                b_price = float(json.loads(resp.read().decode())["price"])

            ts = data.get("ts")
            bar_close = data.get("close")
            dt_str = data.get("datetime", "")[-8:]
            p_down = data.get("prob_down", 0.0)
            p_flat = data.get("prob_flat", 0.0)
            p_up = data.get("prob_up", 0.0)
            direction = data.get("primary_direction")
            dir_name = "DOWN" if direction == 0 else ("FLAT" if direction == 1 else "UP")
            act = data.get("act_probability", 0.0)
            bet = data.get("bet_size", 0.0)
            m_preds = data.get("member_predictions", {})

            is_new = ts not in seen_ts
            if is_new and ts:
                seen_ts.add(ts)
                samples.append({
                    "ts": ts,
                    "bar_close": bar_close,
                    "binance_price": b_price,
                    "direction": direction,
                    "dir_name": dir_name,
                    "p_down": p_down,
                    "p_flat": p_flat,
                    "p_up": p_up,
                    "act": act,
                    "bet": bet,
                    "m_preds": m_preds,
                })

            print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] Bar {dt_str} | Close: ${bar_close:.2f} | Binance Live: ${b_price:.2f} | "
                  f"Dir: {dir_name:4s} (D:{p_down:.2f} F:{p_flat:.2f} U:{p_up:.2f}) | Act: {act:.2f} | Bet: {bet:.2f} | "
                  f"{'NEW BAR' if is_new else 'same bar'}")

            # Print member predictions summary if new bar
            if is_new:
                for m_k, m_v in m_preds.items():
                    print(f"    - {m_k:18s}: D={m_v[0]:.4f}, F={m_v[1]:.4f}, U={m_v[2]:.4f}")

        except Exception as e:
            print(f"Error fetching tick: {e}")

        time.sleep(5)

if __name__ == "__main__":
    main()
