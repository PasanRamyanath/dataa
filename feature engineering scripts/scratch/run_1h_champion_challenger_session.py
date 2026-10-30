"""
feature engineering scripts/scratch/run_1h_champion_challenger_session.py
1-Hour Live Paper Trading Session Runner with 1 Champion + 5 Challengers Tournament Framework.
Ingests live Binance Futures tick data and all 16 neural models simultaneously into 6 competing engines:
  - Champion: Reigning Peak Quad-Alpha Engine
  - Challenger 1: Micro-Price OFI & Volatility Ratchet Harvester
  - Challenger 2: High-Frequency Tactical Impulse Scalper (8-12 tr/hr)
  - Challenger 3: Queue Priority & Micro-Hazard Sniper
  - Challenger 4: Conformal Uncertainty & Shannon Negentropy Stopping Engine
  - Challenger 5: HMM Regime-Switching Adaptive Barrier Engine
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
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FEAT_ENGINE_DIR = (REPO_ROOT / "feature engineering scripts").resolve()
for p in [str(REPO_ROOT), str(FEAT_ENGINE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# 4. Logging configuration
LOG_FILE = REPO_ROOT / "scratch" / "live_champion_challenger.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOG_FILE), mode="w", encoding="utf-8"),
    ]
)
log = logging.getLogger("tournament_runner")

from streaming.run_live_hierarchical_session import LiveHierarchicalTradingSession
from execution.champion_challenger_framework import ChampionChallengerTournamentManager

LEDGER_PATH = REPO_ROOT / "scratch" / "live_hierarchical_session_1h_ledger.jsonl"
LEADERBOARD_PATH = REPO_ROOT / "scratch" / "champion_challenger_leaderboard.json"
HEARTBEAT_PATH = REPO_ROOT / "scratch" / "live_session_1h_heartbeat.json"
FINAL_RESULTS_PATH = REPO_ROOT / "scratch" / "champion_challenger_final_results.json"


class ChampionChallengerTradingSession(LiveHierarchicalTradingSession):
    """
    Unified Tournament Session: feeds real-time ticks and decisions
    to 1 Champion and 5 Challenger execution engines simultaneously.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        bars = int(self.duration_seconds / 10.0)
        self.tournament_mgr = ChampionChallengerTournamentManager(symbol=self.symbol, session_bars=bars)
        self.last_heartbeat_bar = 0

    def start(self) -> None:
        log.info("Bootstrapping multi-horizon ring buffer from Binance REST...")
        stats = self.ring_buffer.bootstrap_from_binance_rest(limit=100)
        log.info("Bootstrap complete across all 14 timeframes: %s", stats)

        # Pre-seed rolling 15m price buffer for all competitor engines to prevent cold-start boundary distortion
        if "1m" in self.ring_buffer.klines_um and len(self.ring_buffer.klines_um["1m"]) >= 15:
            recent_1m = list(self.ring_buffer.klines_um["1m"])[-15:]
            seed_prices = [float(k["close"]) for k in recent_1m]
            for c in self.tournament_mgr.competitors.values():
                for sp in seed_prices:
                    c.engine.rolling_15m_prices.append(sp)
            log.info("Pre-seeded %d 15m price points to all competitor engines (15m Range: $%.2f - $%.2f)",
                     len(seed_prices), min(seed_prices), max(seed_prices))

        log.info("Starting real-time Binance WebSocket stream, REST companion poller & Live News Engine...")
        self.running = True
        if getattr(self.ring_buffer.model_registry, "news_daemon", None) is not None:
            self.ring_buffer.model_registry.news_daemon.start_in_background()
        self.rest_poller.start_in_background()
        self.ws_ingestor.start_in_background()

        self.start_time = time.time()
        log.info("=" * 80)
        log.info("LIVE HIERARCHICAL TRADING & MONITORING SESSION STARTED (%s)", self.symbol)
        log.info("Target Duration: %.1f minutes (%.1f hours)", self.duration_seconds / 60.0, self.duration_seconds / 3600.0)
        log.info("Ledger Output: %s", self.ledger_path)
        log.info("=" * 80)

        try:
            while (time.time() - self.start_time) < self.duration_seconds:
                time.sleep(1.0)
        except KeyboardInterrupt:
            log.info("Session interrupted by user.")
        finally:
            self.stop()

    def on_bar_completed(self, bar_dict: dict) -> None:
        if not self.running:
            return
        self.bars_evaluated += 1
        now_ms = int(bar_dict.get("close_ts", time.time() * 1000))
        c_price = float(bar_dict.get("close", 0.0))
        o_price = float(bar_dict.get("open", c_price))
        h_price = float(bar_dict.get("high", c_price))
        l_price = float(bar_dict.get("low", c_price))
        bbo_bid = float(bar_dict.get("best_bid", c_price))
        bbo_ask = float(bar_dict.get("best_ask", c_price))
        bbo_bid_qty = float(bar_dict.get("bbo_bid_qty", bar_dict.get("bids_depth", 1.0)))
        bbo_ask_qty = float(bar_dict.get("bbo_ask_qty", bar_dict.get("asks_depth", 1.0)))
        bbo_mid = (bbo_bid + bbo_ask) / 2.0 if (bbo_bid > 0 and bbo_ask > 0) else c_price

        sell_vol = float(bar_dict.get("sell_volume", 0.0))
        buy_vol = float(bar_dict.get("buy_volume", 0.0))
        eth_imb = float(bar_dict.get("eth_imbalance", 0.0))

        # 1. Update 14-Timeframe Ring Buffer
        ring_events = self.ring_buffer.push_10s_bar(bar_dict)
        if ring_events.get("15m"):
            log.info(">>> [RingBuffer] 15m Bar Closed | Macro & Meso Probabilities Refreshed")
        elif ring_events.get("5m"):
            log.info(">>> [RingBuffer] 5m Bar Closed | Meso Swing Probabilities Refreshed")

        # 2. Compute dynamic ATRs
        atr_15m = self.ring_buffer.compute_yang_zhang_atr_bps("15m", window=20)
        atr_1h = self.ring_buffer.compute_yang_zhang_atr_bps("1h", window=20)
        atr_10s = max(1.0, atr_15m / 3.5)

        # 3. Predict Micro & Survival
        swing_hint = 2 if (self.ring_buffer.cached_meso_probs[2] >= self.ring_buffer.cached_meso_probs[0]) else 0
        p_micro, micro_safe_fill_prob = self.ring_buffer.model_registry.predict_tier1_micro_and_survival(
            klines_um=self.ring_buffer.klines_um,
            bar_dict=bar_dict,
            primary_dir_hint=swing_hint,
        )

        # 4. Evaluate Multi-Horizon Hierarchical Decision
        decision = self.fusion_engine.evaluate_multi_horizon_tick(
            ts_ms=now_ms,
            p_macro=self.ring_buffer.cached_macro_probs,
            p_meso=self.ring_buffer.cached_meso_probs,
            p_tactical=self.ring_buffer.cached_tactical_probs,
            p_micro=p_micro,
            atr_15m_bps=atr_15m,
            atr_1h_bps=atr_1h,
            atr_10s_bps=atr_10s,
            micro_safe_fill_prob=micro_safe_fill_prob,
            hmm_regime_probs=self.ring_buffer.model_registry.cached_hmm_probs,
        )

        # 5. Broadcast to All 6 Competitor Engines Simultaneously!
        self.tournament_mgr.on_market_micro_tick(
            ts_ms=now_ms,
            bbo_bid=bbo_bid,
            bbo_ask=bbo_ask,
            bbo_bid_qty=bbo_bid_qty,
            bbo_ask_qty=bbo_ask_qty,
            aggressive_sell_vol_at_bid=sell_vol,
            aggressive_buy_vol_at_ask=buy_vol,
            eth_1s_ret_bps=eth_imb * 5.0,
            bar_high=h_price,
            bar_low=l_price,
            current_decision=decision,
        )

        self.tournament_mgr.on_hierarchical_decision(
            decision=decision,
            bbo_bid=bbo_bid,
            bbo_ask=bbo_ask,
            bbo_bid_qty=bbo_bid_qty,
            bbo_ask_qty=bbo_ask_qty,
            equity_usdt=self.equity_usdt,
            intra_bar_open=o_price,
        )

        # 6. Accuracy Tracking
        self.evaluator.record_tick(
            bar_idx=self.bars_evaluated,
            ts_ms=now_ms,
            mid_price=bbo_mid,
            primary_dir=decision.primary_direction,
            macro_probs=decision.macro_probs,
            meso_probs=decision.meso_probs,
            tactical_probs=decision.tactical_probs,
            micro_probs=decision.micro_probs,
        )
        self.evaluator.update_evaluations(self.bars_evaluated, bbo_mid, h_price, l_price)

        # 7. Write to Champion Ledger
        champ_comp = self.tournament_mgr.competitors["Champion"]
        champ_engine = champ_comp.engine
        record = {
            "bar_idx": self.bars_evaluated,
            "ts_ms": now_ms,
            "datetime": datetime.now(timezone.utc).isoformat(),
            "bbo_mid": bbo_mid,
            "bbo_bid": bbo_bid,
            "bbo_ask": bbo_ask,
            "macro_probs": [round(float(x), 4) for x in decision.macro_probs],
            "meso_probs": [round(float(x), 4) for x in decision.meso_probs],
            "tactical_probs": [round(float(x), 4) for x in decision.tactical_probs],
            "micro_probs": [round(float(x), 4) for x in decision.micro_probs],
            "primary_direction": decision.primary_direction,
            "expected_net_edge_bps": round(decision.expected_net_edge_bps, 2),
            "execution_mode": decision.execution_mode,
            "fsm_state": str(getattr(champ_engine, "state", "IDLE")),
            "active_position": champ_engine.active_position.__dict__ if champ_engine.active_position else None,
            "closed_trades_count": len(champ_engine.closed_trades),
            "cumulative_realized_pnl_bps": round(champ_comp.net_pnl_bps, 2),
        }
        with open(LEDGER_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        # 8. Update Leaderboard & Heartbeat every 6 bars (1 minute)
        if self.bars_evaluated - self.last_heartbeat_bar >= 6:
            self.last_heartbeat_bar = self.bars_evaluated
            self._write_tournament_telemetry(bbo_mid)

        # 8. Log Terminal Status
        dir_str = {0: "SHORT", 1: "FLAT", 2: "LONG"}.get(decision.primary_direction, "FLAT")
        champ_comp = self.tournament_mgr.competitors["Champion"]
        log.info(
            "[%s | #%04d] BTC: $%.2f | Dir: %s | Edge: %+.1f bps | Champ Pos: %s | Champ PnL: %+.2f bps",
            datetime.now(timezone.utc).strftime("%H:%M:%S"),
            self.bars_evaluated,
            bbo_mid,
            dir_str,
            decision.expected_net_edge_bps,
            champ_comp.active_position_str,
            champ_comp.net_pnl_bps,
        )

        # 9. Periodic Full Tournament Leaderboard Output (Every 60 bars = 10 minutes)
        if self.bars_evaluated % 60 == 0:
            self._print_tournament_leaderboard(bbo_mid)

        # 10. 15-Minute Monitor Checkpoint Output (Every 90 bars = 15 minutes)
        if self.bars_evaluated % 90 == 0:
            checkpoint_min = int(self.bars_evaluated * 10 / 60)
            log.info("=" * 105)
            log.info(">>> [15-MIN CHECKPOINT] Minute %d / %d (Bar %d) Leaderboard Status:", checkpoint_min, int(self.duration_seconds / 60), self.bars_evaluated)
            self._print_tournament_leaderboard(bbo_mid)
            log.info("=" * 105)

    def _write_tournament_telemetry(self, current_mid: float) -> None:
        try:
            elapsed_sec = time.time() - self.start_time
            acc = self.evaluator.get_summary()
            leaderboard = self.tournament_mgr.update_metrics(current_mid)

            # Save full leaderboard JSON
            with open(LEADERBOARD_PATH, "w", encoding="utf-8") as f:
                json.dump({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "elapsed_minutes": round(elapsed_sec / 60.0, 2),
                    "bars_evaluated": self.bars_evaluated,
                    "current_price": current_mid,
                    "leaderboard": leaderboard,
                }, f, indent=2)

            # Save heartbeat compatible format (anchored to Champion)
            champ = self.tournament_mgr.competitors["Champion"]
            heartbeat = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "elapsed_minutes": round(elapsed_sec / 60.0, 2),
                "target_minutes": round(self.duration_seconds / 60.0, 2),
                "progress_pct": round(min(100.0, elapsed_sec / self.duration_seconds * 100.0), 1),
                "bars_evaluated": self.bars_evaluated,
                "current_price": current_mid,
                "active_position": champ.active_position_str,
                "trades_count": champ.total_trades,
                "wins_count": champ.winning_trades,
                "win_rate_pct": champ.win_rate_pct,
                "net_pnl_bps": champ.net_pnl_bps,
                "accuracy_summary": acc,
                "leaderboard_summary": [
                    {"rank": r["rank"], "name": r["name"], "net_pnl": r["net_pnl_bps"], "wr": r["win_rate_pct"], "trades": r["trades"], "pf": r["profit_factor"]}
                    for r in leaderboard
                ]
            }
            with open(HEARTBEAT_PATH, "w", encoding="utf-8") as f:
                json.dump(heartbeat, f, indent=2)
        except Exception as e:
            log.warning("Tournament telemetry write error: %s", e)

    def _print_tournament_leaderboard(self, current_mid: float) -> None:
        elapsed_min = (time.time() - self.start_time) / 60.0
        leaderboard = self.tournament_mgr.update_metrics(current_mid)

        log.info("=" * 105)
        log.info(">>> TOURNAMENT LEADERBOARD AUDIT (Elapsed: %.1f min | Bars: %d)", elapsed_min, self.bars_evaluated)
        log.info(f"{'Rank':4s} | {'Competitor':14s} | {'Trades':6s} | {'Wins':4s} | {'Win Rate':8s} | {'Net PnL':10s} | {'PF':5s} | {'Active Pos':20s}")
        log.info("-" * 105)
        for r in leaderboard:
            log.info(
                f"#{r['rank']:2d}   | {r['name']:14s} | {r['trades']:6d} | {r['wins']:4d} | {r['win_rate_pct']:7.1f}% | {r['net_pnl_bps']:+9.2f}b | {r['profit_factor']:5.2f} | {r['active_pos']:20s}"
            )
        log.info("=" * 105)

    def stop(self) -> None:
        super().stop()
        # Save final tournament results
        final_leaderboard = self.tournament_mgr.update_metrics(self.evaluator.last_eval_price)
        crown_winner = self.tournament_mgr.determine_tournament_winner(reigning_champ_name="Champion")
        results = {
            "session_duration_minutes": round((time.time() - self.start_time) / 60.0, 2),
            "bars_evaluated": self.bars_evaluated,
            "tournament_winner": crown_winner["name"],
            "winner_profile": crown_winner["strategy_profile"],
            "winner_net_pnl_bps": crown_winner["net_pnl_bps"],
            "winner_win_rate_pct": crown_winner["win_rate_pct"],
            "winner_trades": crown_winner["trades"],
            "final_leaderboard": final_leaderboard,
            "competitor_trade_ledgers": {
                name: [t.__dict__ for t in comp.engine.closed_trades]
                for name, comp in self.tournament_mgr.competitors.items()
            }
        }
        with open(FINAL_RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        log.info("=" * 105)
        log.info("TOURNAMENT CONCLUDED SUCCESSFULLY!")
        log.info("CROWNED CHAMPION: %s (%s)", crown_winner["name"], crown_winner["strategy_profile"])
        log.info("Winning Performance: %+.2f bps Net PnL | %.1f%% Win Rate | %d Trades", crown_winner["net_pnl_bps"], crown_winner["win_rate_pct"], crown_winner["trades"])
        log.info("Final Tournament Results Saved: %s", FINAL_RESULTS_PATH)
        log.info("=" * 105)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Live Champion-Challenger Tournament Session Runner")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol (default: BTCUSDT)")
    parser.add_argument("--duration", type=float, default=120.0, help="Duration in minutes (default: 120.0)")
    parser.add_argument("--equity", type=float, default=600.0, help="Account equity USDT (default: 600.0)")
    args = parser.parse_args()

    log.info("=" * 105)
    log.info("LAUNCHING %.1f-MINUTE CHAMPION-CHALLENGER TOURNAMENT SESSION (1 CHAMPION + 5 CHALLENGERS)", args.duration)
    log.info("Symbol: %s | Target Duration: %.1f minutes | Starting Equity: $%.2f USDT", args.symbol, args.duration, args.equity)
    log.info("=" * 105)

    # Clean previous run artifacts
    for p in [LEADERBOARD_PATH, FINAL_RESULTS_PATH, HEARTBEAT_PATH, LEDGER_PATH]:
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass

    session = ChampionChallengerTradingSession(
        symbol=args.symbol,
        duration_minutes=args.duration,
        equity_usdt=args.equity,
    )
    try:
        session.start()
    except Exception as e:
        log.exception("Fatal error in tournament session: %s", e)
    finally:
        session.stop()

if __name__ == "__main__":
    main()
