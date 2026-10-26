"""
feature engineering scripts/scratch/run_1h_live_paper_session.py
1-Hour Live Paper Trading Session Runner with Real-Time Telemetry and Accuracy Tracking.
Tests updated hierarchical sniper execution engine and forward accuracy evaluator.
"""
import os
import sys
import time
import json
import socket
import logging
from datetime import datetime, timezone
from pathlib import Path

# 1. Force IPv4 routing on Windows to prevent IPv6 DNS/HTTPS connection hangs
old_getaddrinfo = socket.getaddrinfo
def new_getaddrinfo(*args, **kwargs):
    responses = old_getaddrinfo(*args, **kwargs)
    return [r for r in responses if r[0] == socket.AF_INET] or responses
socket.getaddrinfo = new_getaddrinfo

# 2. Configure UTF-8 encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# 3. Path setup
REPO_ROOT = Path("D:/PrivateProjects/BTC Futures").resolve()
FEAT_ENGINE_DIR = (REPO_ROOT / "feature engineering scripts").resolve()
for p in [str(REPO_ROOT), str(FEAT_ENGINE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# 4. Logging configuration
LOG_FILE = REPO_ROOT / "scratch" / "live_paper_1h_updated.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOG_FILE), mode="w", encoding="utf-8"),
    ]
)
log = logging.getLogger("live_1h_runner")

from streaming.run_live_hierarchical_session import LiveHierarchicalTradingSession

LEDGER_PATH = REPO_ROOT / "scratch" / "live_hierarchical_session_1h_ledger.jsonl"
METRICS_PATH = REPO_ROOT / "scratch" / "live_session_1h_final_metrics.json"
HEARTBEAT_PATH = REPO_ROOT / "scratch" / "live_session_1h_heartbeat.json"

class InstrumentedTestSession(LiveHierarchicalTradingSession):
    """
    Subclass that adds periodic heartbeat status export every 6 bars (1 minute)
    for transparent real-time monitoring.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_heartbeat_bar = 0

    def on_bar_completed(self, bar_dict: dict) -> None:
        super().on_bar_completed(bar_dict)
        
        # Periodic heartbeat every 6 bars (~1 minute)
        if self.bars_evaluated - self.last_heartbeat_bar >= 6:
            self.last_heartbeat_bar = self.bars_evaluated
            self._write_heartbeat()

    def _write_heartbeat(self) -> None:
        try:
            elapsed_sec = time.time() - self.start_time
            acc = self.evaluator.get_summary()
            trades = self.sniper_engine.closed_trades
            n_trades = len(trades)
            wins = sum(1 for t in trades if t.net_return_bps > 0)
            win_rate = (wins / n_trades * 100.0) if n_trades > 0 else 0.0
            net_pnl = sum(t.net_return_bps for t in trades)
            
            # Position details
            pos_dict = None
            if self.sniper_engine.active_position:
                ap = self.sniper_engine.active_position
                c_price = self.evaluator.last_eval_price
                unreal_bps = (
                    ((c_price - ap.entry_price) / ap.entry_price) * 10000.0
                    if ap.side == "BUY"
                    else ((ap.entry_price - c_price) / ap.entry_price) * 10000.0
                )
                pos_dict = {
                    "side": ap.side,
                    "entry_price": ap.entry_price,
                    "tp_price": ap.tp_price,
                    "sl_price": ap.sl_price,
                    "unrealized_pnl_bps": round(unreal_bps, 2),
                    "holding_sec": int((time.time() * 1000 - ap.entry_ts_ms) / 1000),
                }

            heartbeat = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "elapsed_minutes": round(elapsed_sec / 60.0, 2),
                "target_minutes": round(self.duration_seconds / 60.0, 2),
                "progress_pct": round(min(100.0, elapsed_sec / self.duration_seconds * 100.0), 1),
                "bars_evaluated": self.bars_evaluated,
                "current_price": self.evaluator.last_eval_price,
                "active_position": pos_dict,
                "trades_count": n_trades,
                "wins_count": wins,
                "win_rate_pct": round(win_rate, 2),
                "net_pnl_bps": round(net_pnl, 2),
                "cached_tier_probs": {
                    "macro": [round(float(x), 4) for x in self.ring_buffer.cached_macro_probs],
                    "meso": [round(float(x), 4) for x in self.ring_buffer.cached_meso_probs],
                    "tactical": [round(float(x), 4) for x in self.ring_buffer.cached_tactical_probs],
                },
                "accuracy_summary": acc,
            }
            with open(HEARTBEAT_PATH, "w", encoding="utf-8") as f:
                json.dump(heartbeat, f, indent=2)
        except Exception as e:
            log.warning("Heartbeat write error: %s", e)

def main():
    log.info("=" * 80)
    log.info("LAUNCHING 1-HOUR LIVE PAPER TRADING & ACCURACY EVALUATION SESSION (V3.4 UPDATED ENGINE)")
    log.info("Symbol: BTCUSDT | Target Duration: 60.0 minutes (1.0 hour)")
    log.info("Checkpoints: %s", FEAT_ENGINE_DIR / "models_v3" / "checkpoints")
    log.info("Ledger Path: %s", LEDGER_PATH)
    log.info("Metrics Path: %s", METRICS_PATH)
    log.info("Heartbeat Path: %s", HEARTBEAT_PATH)
    log.info("=" * 80)

    # Clean previous run artifacts
    for p in [LEDGER_PATH, METRICS_PATH, HEARTBEAT_PATH]:
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass

    session = InstrumentedTestSession(
        symbol="BTCUSDT",
        duration_minutes=60.0,
        ledger_path=LEDGER_PATH,
        equity_usdt=10000.0,
    )
    
    try:
        session.start()
    except Exception as e:
        log.exception("Fatal error during session execution: %s", e)
    finally:
        session.stop()
        # Ensure final metrics saved
        if not METRICS_PATH.exists():
            elapsed_sec = time.time() - session.start_time
            acc_summary = session.evaluator.get_summary()
            trades = session.sniper_engine.closed_trades
            n_trades = len(trades)
            wins = sum(1 for t in trades if t.net_return_bps > 0)
            win_rate = (wins / n_trades * 100.0) if n_trades > 0 else 0.0
            net_pnl_bps = sum(t.net_return_bps for t in trades)
            avg_pnl_bps = (net_pnl_bps / n_trades) if n_trades > 0 else 0.0
            final_metrics = {
                "session_duration_minutes": round(elapsed_sec / 60.0, 2),
                "bars_evaluated": session.bars_evaluated,
                "total_trades": n_trades,
                "winning_trades": wins,
                "win_rate_pct": round(win_rate, 2),
                "total_net_pnl_bps": round(net_pnl_bps, 2),
                "avg_net_pnl_bps": round(avg_pnl_bps, 2),
                "accuracy_summary": acc_summary,
                "closed_trades": [t.__dict__ for t in trades],
            }
            with open(METRICS_PATH, "w", encoding="utf-8") as f:
                json.dump(final_metrics, f, indent=2)

        log.info("1-Hour session finished. Exiting process.")
        # Force terminate all background WebSocket / REST threads
        os._exit(0)

if __name__ == "__main__":
    main()
