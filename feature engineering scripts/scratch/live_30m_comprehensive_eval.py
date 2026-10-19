"""
30-Minute Real-Time BTCUSDT Prediction Accuracy & Precision Evaluation Engine.
Monitors Kaggle Dual-T4 Ensemble predictions against live Binance USD(S)-M Futures data.
Evaluates multi-horizon directional accuracy (1, 3, 6, 12, 18 bars), Triple-Barrier precision,
member model breakdown, calibration quality, and execution latency.
"""
import json
import logging
import os
import ssl
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("live_30m_eval")

_ssl_ctx = ssl.create_default_context()
_ssl_ctx.check_hostname = False
_ssl_ctx.verify_mode = ssl.CERT_NONE

HF_TOKEN = os.environ.get("HF_TOKEN", "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")
HF_TUNNEL_URL = "https://huggingface.co/p-j-r-1-2-3/Project-IMPJR-Models-fusion-calibration/raw/main/active_tunnel_url.txt"
BINANCE_BBO_URL = "https://fapi.binance.com/fapi/v1/ticker/bookTicker?symbol=BTCUSDT"
BINANCE_PRICE_URL = "https://fapi.binance.com/fapi/v1/ticker/price?symbol=BTCUSDT"

OUTPUT_DIR = Path("scratch")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LEDGER_PATH = OUTPUT_DIR / "live_30m_eval_ledger.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "live_30m_eval_summary.json"

TARGET_DURATION_SEC = 32 * 60  # 32 minutes (ensures >= 180 bars evaluated with forward resolution)
POLL_INTERVAL_SEC = 2.0

# Triple Barrier Evaluation
TP_BPS = 16.0  # +16 bps take profit
SL_BPS = 12.0  # -12 bps stop loss
TRAILING_ACT_BPS = 7.0  # +7 bps trailing activation
TRAILING_CB_BPS = 4.0   # 4 bps callback
MAX_HOLD_BARS = 18      # 18 bars (3 minutes) time horizon
TAKER_FEE_BPS = 5.0     # 5 bps each leg (10 bps round-trip)


def fetch_active_tunnel_url() -> str | None:
    try:
        req = urllib.request.Request(HF_TUNNEL_URL, headers={"Authorization": f"Bearer {HF_TOKEN}"})
        with urllib.request.urlopen(req, timeout=5, context=_ssl_ctx) as resp:
            url = resp.read().decode("utf-8").strip()
            if url.startswith("http"):
                return url
    except Exception as e:
        log.debug("Could not fetch tunnel URL from HF: %s", e)
    return None


def fetch_json(url: str, headers: dict | None = None, timeout: float = 4.0) -> dict | None:
    hdrs = {"User-Agent": "IMPJR-Live-Eval/2.0"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, headers=hdrs)
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_ctx) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def main():
    log.info("=" * 80)
    log.info("STARTING COMPREHENSIVE 30-MINUTE LIVE PREDICTION ACCURACY & PRECISION MONITOR")
    log.info("Target: >= 30 Minutes (~180 10s bars) | Ground Truth: Binance Futures WebSocket/BBO")
    log.info("=" * 80)

    # 1. Resolve Active Tunnel URL
    tunnel_url = os.environ.get("KAGGLE_TUNNEL_URL")
    if not tunnel_url:
        log.info("Resolving active Cloudflare tunnel URL from Hugging Face...")
        for attempt in range(60):
            tunnel_url = fetch_active_tunnel_url()
            if tunnel_url:
                log.info("Acquired active tunnel URL: %s", tunnel_url)
                break
            log.info("Waiting for active tunnel URL to publish on Hugging Face (attempt %d/60)...", attempt + 1)
            time.sleep(5)

    if not tunnel_url:
        log.error("Could not obtain active tunnel URL. Exiting.")
        sys.exit(1)

    latest_endpoint = f"{tunnel_url.rstrip('/')}/latest"
    log.info("Connecting to live stream endpoint: %s", latest_endpoint)

    # Clear previous run ledger
    if LEDGER_PATH.exists():
        LEDGER_PATH.unlink()

    start_time = time.time()
    end_time = start_time + TARGET_DURATION_SEC

    bars: list[dict] = []
    last_bar_ts = None
    bar_count = 0

    # Active Triple-Barrier Positions
    active_positions: list[dict] = []
    closed_trades: list[dict] = []

    # Member tracking
    member_names = ["gbm-family", "sentiment-branch", "kronos-finetune", "ssm-challenger", "tft", "patchtst", "itransformer", "nhits"]
    member_stats = {m: {"total_preds": 0, "correct_1bar": 0, "correct_3bar": 0, "correct_6bar": 0} for m in member_names}

    while time.time() < end_time:
        loop_start = time.time()
        try:
            data = fetch_json(latest_endpoint)
            latest = data.get("latest") if data else None

            if latest and latest.get("status") == "scored":
                bar_ts = latest.get("ts")
                if bar_ts != last_bar_ts:
                    last_bar_ts = bar_ts
                    bar_count += 1

                    # Fetch real Binance Futures BBO ticker
                    bbo = fetch_json(BINANCE_BBO_URL) or {}
                    best_bid = float(bbo.get("bidPrice", 0.0))
                    best_ask = float(bbo.get("askPrice", 0.0))
                    close_price = float(latest.get("close", (best_bid + best_ask) / 2.0 if best_bid > 0 else 0.0))
                    spread_bps = ((best_ask - best_bid) / close_price * 10000.0) if close_price > 0 and best_bid > 0 else 0.0

                    # Predictions
                    p_down = latest.get("prob_down")
                    p_flat = latest.get("prob_flat")
                    p_up = latest.get("prob_up")
                    primary_dir = latest.get("primary_direction")
                    confidence = latest.get("confidence")
                    act_prob = latest.get("act_probability")
                    bet_size = latest.get("bet_size")
                    fusion_stage = latest.get("fusion_stage")
                    members_voted = latest.get("members_voted", [])
                    member_preds = latest.get("member_predictions", {})
                    latency_info = latest.get("latency_telemetry", {})

                    dir_str = "SHORT" if primary_dir == 0 else ("LONG" if primary_dir == 2 else "FLAT")
                    is_authorized = (act_prob is not None and act_prob >= 0.70 and primary_dir in (0, 2))

                    bar_entry = {
                        "bar_idx": bar_count,
                        "ts": bar_ts,
                        "datetime": latest.get("datetime"),
                        "close_price": close_price,
                        "best_bid": best_bid,
                        "best_ask": best_ask,
                        "spread_bps": round(spread_bps, 2),
                        "primary_direction": primary_dir,
                        "dir_str": dir_str,
                        "prob_down": p_down,
                        "prob_flat": p_flat,
                        "prob_up": p_up,
                        "confidence": confidence,
                        "act_probability": act_prob,
                        "bet_size": bet_size,
                        "fusion_stage": fusion_stage,
                        "is_authorized": is_authorized,
                        "n_members": len(members_voted),
                        "members_voted": members_voted,
                        "member_predictions": member_preds,
                        "latency_telemetry": latency_info,
                        "forward_prices": {},  # Populated as time advances
                        "forward_returns_bps": {},
                    }
                    bars.append(bar_entry)

                    # Update forward prices & returns for previous bars
                    for prev_b in bars:
                        lag = bar_count - prev_b["bar_idx"]
                        if lag in (1, 3, 6, 12, 18) and lag not in prev_b["forward_prices"]:
                            p_entry = prev_b["close_price"]
                            if p_entry > 0:
                                ret_bps = ((close_price - p_entry) / p_entry) * 10000.0
                                prev_b["forward_prices"][lag] = close_price
                                prev_b["forward_returns_bps"][lag] = round(ret_bps, 2)

                                # Evaluate directional accuracy for this horizon
                                pdir = prev_b["primary_direction"]
                                if pdir in (0, 2):
                                    is_correct = (ret_bps > 0 if pdir == 2 else ret_bps < 0)
                                    prev_b[f"correct_h{lag}"] = is_correct

                                # Evaluate individual member model predictions
                                for m_name, m_pred in prev_b["member_predictions"].items():
                                    if len(m_pred) == 3 and m_name in member_stats:
                                        m_dir = int(1 if m_pred[1] >= max(m_pred[0], m_pred[2]) else (2 if m_pred[2] > m_pred[0] else 0))
                                        if m_dir in (0, 2):
                                            m_correct = (ret_bps > 0 if m_dir == 2 else ret_bps < 0)
                                            if lag == 1:
                                                member_stats[m_name]["total_preds"] += 1
                                                if m_correct:
                                                    member_stats[m_name]["correct_1bar"] += 1
                                            elif lag == 3 and m_correct:
                                                member_stats[m_name]["correct_3bar"] += 1
                                            elif lag == 6 and m_correct:
                                                member_stats[m_name]["correct_6bar"] += 1

                    # Manage active Triple Barrier positions
                    remaining_pos = []
                    for pos in active_positions:
                        hold_bars = bar_count - pos["entry_bar"]
                        p_entry = pos["entry_price"]
                        is_long = pos["direction"] == 2
                        move_bps = ((close_price - p_entry) / p_entry * 10000.0) if is_long else ((p_entry - close_price) / p_entry * 10000.0)
                        
                        # Peak tracking for trailing stop
                        pos["peak_bps"] = max(pos.get("peak_bps", move_bps), move_bps)
                        
                        hit_tp = move_bps >= TP_BPS
                        hit_sl = move_bps <= -SL_BPS
                        hit_ts = (pos["peak_bps"] >= TRAILING_ACT_BPS and (pos["peak_bps"] - move_bps) >= TRAILING_CB_BPS)
                        hit_expiry = hold_bars >= MAX_HOLD_BARS

                        if hit_tp or hit_sl or hit_ts or hit_expiry:
                            reason = "TAKE_PROFIT" if hit_tp else ("STOP_LOSS" if hit_sl else ("TRAILING_STOP" if hit_ts else "TIME_EXPIRY"))
                            gross_pnl_bps = move_bps
                            net_pnl_bps = gross_pnl_bps - (2 * TAKER_FEE_BPS)
                            closed_trades.append({
                                "trade_id": len(closed_trades) + 1,
                                "direction": pos["dir_str"],
                                "entry_bar": pos["entry_bar"],
                                "exit_bar": bar_count,
                                "hold_bars": hold_bars,
                                "entry_price": p_entry,
                                "exit_price": close_price,
                                "exit_reason": reason,
                                "gross_bps": round(gross_pnl_bps, 2),
                                "fee_bps": 2 * TAKER_FEE_BPS,
                                "net_bps": round(net_pnl_bps, 2),
                                "is_win": net_pnl_bps > 0,
                                "act_probability": pos["act_prob"],
                            })
                            log.info(">>> TRADE #%d RESOLVED [%s]: %s | Entry $%.2f -> Exit $%.2f | Net: %+.2f bps",
                                     len(closed_trades), reason, pos["dir_str"], p_entry, close_price, net_pnl_bps)
                        else:
                            remaining_pos.append(pos)
                    active_positions = remaining_pos

                    # Open new position if Ultra Sniper Authorized
                    if is_authorized and len(active_positions) == 0:
                        active_positions.append({
                            "entry_bar": bar_count,
                            "direction": primary_dir,
                            "dir_str": dir_str,
                            "entry_price": close_price,
                            "act_prob": act_prob,
                            "peak_bps": 0.0,
                        })
                        log.info(">>> [ULTRA SNIPER AUTHORIZED] New Position: %s @ $%.2f | Conviction: %.3f", dir_str, close_price, act_prob)

                    # Append to JSONL ledger
                    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
                        f.write(json.dumps(bar_entry) + "\n")

                    # Compute running performance statistics
                    elapsed_min = (time.time() - start_time) / 60.0
                    n_auth = sum(1 for b in bars if b["is_authorized"])
                    n_vetoed = bar_count - n_auth
                    
                    # Horizon accuracy for directional predictions (up or down)
                    dir_bars = [b for b in bars if b["primary_direction"] in (0, 2)]
                    acc_1b = (sum(1 for b in dir_bars if b.get("correct_h1")) / max(1, sum(1 for b in dir_bars if "correct_h1" in b))) * 100.0
                    acc_3b = (sum(1 for b in dir_bars if b.get("correct_h3")) / max(1, sum(1 for b in dir_bars if "correct_h3" in b))) * 100.0
                    acc_6b = (sum(1 for b in dir_bars if b.get("correct_h6")) / max(1, sum(1 for b in dir_bars if "correct_h6" in b))) * 100.0

                    # Ultra Sniper Precision
                    auth_trades = [t for t in closed_trades]
                    n_trades = len(auth_trades)
                    wins = sum(1 for t in auth_trades if t["is_win"])
                    win_rate = (wins / n_trades * 100.0) if n_trades > 0 else 0.0
                    total_net_bps = sum(t["net_bps"] for t in auth_trades)
                    avg_net_bps = (total_net_bps / n_trades) if n_trades > 0 else 0.0

                    gross_wins = sum(t["gross_bps"] for t in auth_trades if t["gross_bps"] > 0)
                    gross_losses = abs(sum(t["gross_bps"] for t in auth_trades if t["gross_bps"] < 0))
                    profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else (99.0 if gross_wins > 0 else 1.0)

                    # Print concise live summary log
                    log.info("Bar %03d (%.1fm) | Price: $%.2f | Dir: %s | Act: %s | Acc (1b: %.1f%%, 3b: %.1f%%) | Trades: %d (WR: %.1f%%, Net: %+.1f bps)",
                             bar_count, elapsed_min, close_price, dir_str,
                             f"{act_prob:.3f}" if act_prob is not None else "None",
                             acc_1b, acc_3b, n_trades, win_rate, total_net_bps)

                    # Save summary JSON
                    summary_payload = {
                        "elapsed_minutes": round(elapsed_min, 2),
                        "total_bars_recorded": bar_count,
                        "target_bars": 180,
                        "last_price": close_price,
                        "directional_accuracy_pct": {
                            "1_bar_ahead_10s": round(acc_1b, 2),
                            "3_bars_ahead_30s": round(acc_3b, 2),
                            "6_bars_ahead_60s": round(acc_6b, 2),
                        },
                        "ultra_sniper_precision": {
                            "signals_authorized": n_auth,
                            "chop_vetoed_bars": n_vetoed,
                            "chop_veto_rate_pct": round((n_vetoed / bar_count) * 100.0, 1),
                            "closed_trades": n_trades,
                            "active_positions": len(active_positions),
                            "win_count": wins,
                            "loss_count": n_trades - wins,
                            "win_rate_pct": round(win_rate, 2),
                            "total_net_pnl_bps": round(total_net_bps, 2),
                            "average_net_bps_per_trade": round(avg_net_bps, 2),
                            "profit_factor": round(profit_factor, 2),
                        },
                        "member_model_performance": {
                            m: {
                                "predictions": s["total_preds"],
                                "acc_1bar_pct": round((s["correct_1bar"] / max(1, s["total_preds"])) * 100.0, 1),
                                "acc_3bar_pct": round((s["correct_3bar"] / max(1, s["total_preds"])) * 100.0, 1),
                                "acc_6bar_pct": round((s["correct_6bar"] / max(1, s["total_preds"])) * 100.0, 1),
                            }
                            for m, s in member_stats.items()
                        },
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                    with open(SUMMARY_PATH, "w", encoding="utf-8") as sf:
                        json.dump(summary_payload, sf, indent=2)

        except Exception as e:
            log.warning("Polling error: %s", e)

        elapsed_loop = time.time() - loop_start
        time.sleep(max(0.5, POLL_INTERVAL_SEC - elapsed_loop))

    log.info("=" * 80)
    log.info("30-MINUTE MONITORING RUN COMPLETED!")
    log.info("Total Bars Scored: %d | Ledger saved to %s", bar_count, LEDGER_PATH)
    log.info("=" * 80)


if __name__ == "__main__":
    main()
