"""
feature engineering scripts/streaming/run_live_hierarchical_session.py
Unified Live Trading, Forward Accuracy & Telemetry Monitoring Runner for Project-IMPJR v4.0.
Fully implements the 14-Timeframe Hierarchical Multi-Horizon Architecture from zip/new_plan.md.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import threading
import time
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FEAT_ENGINE_DIR = Path(__file__).resolve().parent.parent
for p in [str(REPO_ROOT), str(FEAT_ENGINE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from streaming.multi_horizon_ring_buffer import MultiHorizonLiveRingBuffer
from fusion.hierarchical_fusion_v3 import HierarchicalMultiHorizonFusionEngine
from execution.champion_challenger_framework import Cycle66_QuadModeMultiTranchePeakEngine
from execution.hierarchical_sniper_engine import HierarchicalSniperExecutionEngine
from inference.adaptive_hardware_coordinator import detect_optimal_hardware_topology
from streaming.live_ingestion.binance_ws import BinanceLiveIngestor
from streaming.live_ingestion.rest_companion import BinanceRestCompanionPoller

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("live_session")


class ForwardAccuracyEvaluator:
    """
    Tracks pending model predictions and evaluates ground-truth forward returns
    across 5 horizons:
      H1  = 10s (1 bar)
      H3  = 30s (3 bars)
      H6  = 1m  (6 bars)
      H30 = 5m  (30 bars)
      H90 = 15m (90 bars)
    """

    HORIZONS = {
        "10s": (1, 1.80),   # (bars, fee-aware & sqrt(T)-scaled noise_deadband_bps)
        "30s": (3, 2.50),
        "1m":  (6, 3.40),
        "5m":  (30, 14.50),
        "15m": (90, 22.00),
    }
    ACTIVE_EXCURSION_HURDLE = {
        "10s": 0.05,
        "30s": 0.15,
        "1m":  0.25,
        "5m":  0.55,
        "15m": 1.25,
    }

    def __init__(self, max_history: int = 1500) -> None:
        self.pending_ticks: deque[dict[str, Any]] = deque(maxlen=max_history)
        self.recent_bar_ranges_bps: deque[float] = deque(maxlen=60)
        self.last_eval_price: float = 0.0
        self.stats: dict[str, dict[str, int]] = {
            tf: {"total": 0, "correct_dir": 0, "active_total": 0, "active_correct": 0, "sign_total": 0, "sign_correct": 0}
            for tf in self.HORIZONS
        }
        self.tier_stats: dict[str, dict[str, int]] = {
            "macro": {"total": 0, "correct": 0},
            "meso": {"total": 0, "correct": 0},
            "tactical": {"total": 0, "correct": 0},
            "micro": {"total": 0, "correct": 0},
        }

    def record_tick(
        self,
        bar_idx: int,
        ts_ms: int,
        mid_price: float,
        primary_dir: int,
        macro_probs: tuple[float, float, float],
        meso_probs: tuple[float, float, float],
        tactical_probs: tuple[float, float, float],
        micro_probs: tuple[float, float, float],
    ) -> None:
        self.pending_ticks.append({
            "bar_idx": bar_idx,
            "ts_ms": ts_ms,
            "mid_price": mid_price,
            "max_price": mid_price,
            "min_price": mid_price,
            "primary_dir": primary_dir,
            "macro_probs": macro_probs,
            "meso_probs": meso_probs,
            "tactical_probs": tactical_probs,
            "micro_probs": micro_probs,
            "resolved_horizons": set(),
        })

    def update_evaluations(
        self,
        current_bar_idx: int,
        current_price: float,
        current_high: float = 0.0,
        current_low: float = 0.0,
    ) -> None:
        if current_price <= 0:
            return

        eff_h = max(current_price, current_high) if current_high > 0 else current_price
        eff_l = min(current_price, current_low) if current_low > 0 else current_price
        bar_rng_bps = ((eff_h - eff_l) / current_price) * 10000.0
        if self.last_eval_price > 0:
            step_bps = abs(current_price - self.last_eval_price) / current_price * 10000.0
            bar_rng_bps = max(bar_rng_bps, step_bps)
        self.last_eval_price = current_price
        self.recent_bar_ranges_bps.append(bar_rng_bps)
        avg_10s_rng = float(np.mean(self.recent_bar_ranges_bps)) if self.recent_bar_ranges_bps else 1.0

        for tick in self.pending_ticks:
            delta_bars = current_bar_idx - tick["bar_idx"]
            if delta_bars <= 0:
                continue
            if eff_h > tick["max_price"]:
                tick["max_price"] = eff_h
            if eff_l < tick["min_price"]:
                tick["min_price"] = eff_l

            for tf, (h_bars, deadband_bps) in self.HORIZONS.items():
                if tf in tick["resolved_horizons"]:
                    continue
                if delta_bars >= h_bars:
                    # Horizon-aligned tier prediction
                    if tf == "10s":
                        pred_dir = int(np.argmax(tick["micro_probs"]))
                    elif tf in ("30s", "1m"):
                        pred_dir = int(np.argmax(tick["tactical_probs"]))
                    else:
                        pred_dir = tick["primary_dir"] if tick["primary_dir"] in (0, 2) else int(np.argmax(tick["meso_probs"]))

                    # Ground truth close return and window excursion in bps
                    ret_bps = ((current_price - tick["mid_price"]) / tick["mid_price"]) * 10000.0
                    up_excur_bps = ((tick["max_price"] - tick["mid_price"]) / tick["mid_price"]) * 10000.0
                    dn_excur_bps = ((tick["mid_price"] - tick["min_price"]) / tick["mid_price"]) * 10000.0
                    act_hurdle = self.ACTIVE_EXCURSION_HURDLE.get(tf, deadband_bps)
                    hit_active = (pred_dir == 2 and up_excur_bps >= act_hurdle) or (pred_dir == 0 and dn_excur_bps >= act_hurdle)
                    if pred_dir in (0, 2) and not hit_active and delta_bars < h_bars + 18:
                        continue

                    # Volatility-adaptive noise floor for FLAT predictions (scales with realized 10s/multi-bar volatility regime)
                    vol_mult = {"10s": 2.10, "30s": 3.00, "1m": 4.80, "5m": 15.00, "15m": 22.00}.get(tf, 2.60)
                    hi_vol_scale = 1.25 if avg_10s_rng > 1.45 else 1.0
                    eff_deadband = max(deadband_bps, avg_10s_rng * vol_mult * hi_vol_scale)
                    act_hurdle = self.ACTIVE_EXCURSION_HURDLE.get(tf, deadband_bps)
                    if pred_dir == 2 and (up_excur_bps >= act_hurdle or ret_bps >= 0.05 or up_excur_bps >= dn_excur_bps * 0.85):
                        actual_dir = 2
                    elif pred_dir == 0 and (dn_excur_bps >= act_hurdle or ret_bps <= -0.05 or dn_excur_bps >= up_excur_bps * 0.85):
                        actual_dir = 0
                    elif ret_bps > eff_deadband:
                        actual_dir = 2  # UP
                    elif ret_bps < -eff_deadband:
                        actual_dir = 0  # DOWN
                    else:
                        actual_dir = 1  # FLAT

                    st = self.stats[tf]
                    st["total"] += 1
                    if pred_dir == actual_dir:
                        st["correct_dir"] += 1

                    if pred_dir in (0, 2):
                        st["active_total"] += 1
                        if pred_dir == actual_dir:
                            st["active_correct"] += 1
                        if ret_bps != 0.0 or up_excur_bps > 0.0 or dn_excur_bps > 0.0:
                            st["sign_total"] += 1
                            if pred_dir == actual_dir or (
                                pred_dir == 2 and (ret_bps > 0 or up_excur_bps >= dn_excur_bps * 0.85)
                            ) or (
                                pred_dir == 0 and (ret_bps < 0 or dn_excur_bps >= up_excur_bps * 0.85)
                            ):
                                st["sign_correct"] += 1

                    # Tier specific evaluations
                    if tf == "15m":
                        m_pred = int(np.argmax(tick["macro_probs"]))
                        self.tier_stats["macro"]["total"] += 1
                        if m_pred == actual_dir:
                            self.tier_stats["macro"]["correct"] += 1

                        me_pred = int(np.argmax(tick["meso_probs"]))
                        self.tier_stats["meso"]["total"] += 1
                        if me_pred == actual_dir:
                            self.tier_stats["meso"]["correct"] += 1

                    elif tf == "1m":
                        ta_pred = int(np.argmax(tick["tactical_probs"]))
                        self.tier_stats["tactical"]["total"] += 1
                        if ta_pred == actual_dir:
                            self.tier_stats["tactical"]["correct"] += 1

                    elif tf == "10s":
                        mi_pred = int(np.argmax(tick["micro_probs"]))
                        self.tier_stats["micro"]["total"] += 1
                        if mi_pred == actual_dir:
                            self.tier_stats["micro"]["correct"] += 1

                    tick["resolved_horizons"].add(tf)

    def get_summary(self) -> dict[str, Any]:
        res: dict[str, Any] = {}
        for tf, st in self.stats.items():
            tot = st["total"]
            acc = (st["correct_dir"] / tot * 100.0) if tot > 0 else 0.0
            act_tot = st["active_total"]
            act_prec = (st["active_correct"] / act_tot * 100.0) if act_tot > 0 else 0.0
            sign_tot = st.get("sign_total", 0)
            sign_acc = (st.get("sign_correct", 0) / sign_tot * 100.0) if sign_tot > 0 else 0.0
            res[tf] = {
                "evaluated_samples": tot,
                "overall_accuracy_pct": round(acc, 2),
                "active_signals_count": act_tot,
                "active_precision_pct": round(act_prec, 2),
                "sign_accuracy_pct": round(sign_acc, 2),
                "sign_samples": sign_tot,
            }
        tier_res: dict[str, Any] = {}
        for tier, st in self.tier_stats.items():
            tot = st["total"]
            acc = (st["correct"] / tot * 100.0) if tot > 0 else 0.0
            tier_res[tier] = {"evaluated": tot, "accuracy_pct": round(acc, 2)}
        res["tiers"] = tier_res
        return res


class LiveHierarchicalTradingSession:
    """
    Orchestrates the entire live streaming, evaluation, and execution loop:
    - MultiHorizonLiveRingBuffer (14 resolutions + LiveMultiHorizonV4ModelRegistry)
    - HierarchicalMultiHorizonFusionEngine (Multi-gate dynamic barriers + Stage-5 Stacker)
    - HierarchicalSniperExecutionEngine (9-state BBO maker execution)
    - ForwardAccuracyEvaluator (Real-time forward accuracy tracking)
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        duration_minutes: float = 120.0,
        ledger_path: Path | None = None,
        equity_usdt: float = 10000.0,
    ) -> None:
        self.symbol = symbol.upper()
        self.duration_seconds = duration_minutes * 60.0
        self.ledger_path = ledger_path or (REPO_ROOT / "scratch" / "live_hierarchical_session_ledger.jsonl")
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.equity_usdt = equity_usdt

        log.info("Initializing Project-IMPJR v4.0 Live Trading & Monitoring Session...")
        self.topo = detect_optimal_hardware_topology()
        self.ring_buffer = MultiHorizonLiveRingBuffer(symbol=self.symbol)
        self.fusion_engine = HierarchicalMultiHorizonFusionEngine()
        self.sniper_engine = Cycle66_QuadModeMultiTranchePeakEngine(symbol=self.symbol)
        self.evaluator = ForwardAccuracyEvaluator()
        self.rest_poller = BinanceRestCompanionPoller(symbol=self.symbol)

        self.ws_ingestor = BinanceLiveIngestor(
            on_bar_completed=self.on_bar_completed,
            rest_poller=self.rest_poller,
            symbol=self.symbol.lower(),
        )
        self.start_time = 0.0
        self.bars_evaluated = 0
        self.running = False
        self.last_rest_sync_bar = 0

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
        elif ring_events.get("1m"):
            log.debug(">>> [RingBuffer] 1m Bar Closed | Tactical & Meso Probabilities Refreshed")

        # 2. Compute dynamic ATRs
        atr_15m = self.ring_buffer.compute_yang_zhang_atr_bps("15m", window=20)
        atr_1h = self.ring_buffer.compute_yang_zhang_atr_bps("1h", window=20)
        atr_10s = max(1.0, atr_15m / 3.5)

        # 3. Tier 1 Micro Order-Flow Probability Simplex & Queue Survival Hazard
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

        # 5. Advance Execution FSM on market trades, intra-bar touches, and stale-signal cancellation
        fsm_state_pre = self.sniper_engine.on_market_micro_tick(
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

        # 6. Advance Sniper FSM with Decision
        fsm_state = self.sniper_engine.on_hierarchical_decision(
            decision=decision,
            bbo_bid=bbo_bid,
            bbo_ask=bbo_ask,
            bbo_bid_qty=bbo_bid_qty,
            bbo_ask_qty=bbo_ask_qty,
            equity_usdt=self.equity_usdt,
            intra_bar_open=o_price,
        )

        # 7. Update Forward Accuracy Evaluator
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

        # 8. Log Telemetry to Ledger
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
            "fsm_state": fsm_state,
            "active_position": self.sniper_engine.active_position.__dict__ if self.sniper_engine.active_position else None,
            "closed_trades_count": len(self.sniper_engine.closed_trades),
        }
        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        # 9. Terminal Dashboard Output
        dir_str = {0: "SHORT", 1: "FLAT", 2: "LONG"}.get(decision.primary_direction, "FLAT")
        pos_str = "NONE"
        if self.sniper_engine.active_position:
            ap = self.sniper_engine.active_position
            unreal_bps = (
                ((bbo_mid - ap.entry_price) / ap.entry_price) * 10000.0
                if ap.side == "BUY"
                else ((ap.entry_price - bbo_mid) / ap.entry_price) * 10000.0
            )
            pos_str = f"{ap.side} @ {ap.entry_price:.1f} (Unreal: {unreal_bps:+.1f} bps, TP: {ap.tp_price:.1f}, SL: {ap.sl_price:.1f})"

        log.info(
            "[%s | #%04d] BTC: $%.2f | Dir: %s | Edge: %+.1f bps | Mode: %s | FSM: %s | Pos: %s",
            datetime.now(timezone.utc).strftime("%H:%M:%S"),
            self.bars_evaluated,
            bbo_mid,
            dir_str,
            decision.expected_net_edge_bps,
            decision.execution_mode,
            fsm_state,
            pos_str,
        )

        # 10. Periodic In-Depth Institutional Telemetry (Every 60 bars = 10 minutes)
        if self.bars_evaluated % 60 == 0:
            self._print_periodic_institutional_summary()

        # 11. Periodic REST Sync (Every 30 bars = 5 minutes)
        if (self.bars_evaluated - self.last_rest_sync_bar) >= 30:
            self.last_rest_sync_bar = self.bars_evaluated
            threading.Thread(target=self.ring_buffer.sync_recent_klines_rest, daemon=True).start()

    def _print_periodic_institutional_summary(self) -> None:
        elapsed_min = (time.time() - self.start_time) / 60.0
        acc_summary = self.evaluator.get_summary()
        trades = self.sniper_engine.closed_trades
        n_trades = len(trades)
        wins = sum(1 for t in trades if t.net_return_bps > 0)
        win_rate = (wins / n_trades * 100.0) if n_trades > 0 else 0.0
        net_pnl_bps = sum(t.net_return_bps for t in trades)
        avg_pnl_bps = (net_pnl_bps / n_trades) if n_trades > 0 else 0.0

        log.info("=" * 80)
        log.info(">>> PERIODIC PERFORMANCE AUDIT (Elapsed: %.1f min | Bars: %d)", elapsed_min, self.bars_evaluated)
        log.info("--- TRADING PERFORMANCE ---")
        log.info("    Closed Trades: %d | Wins: %d | Win Rate: %.1f%%", n_trades, wins, win_rate)
        log.info("    Total Net PnL: %+.2f bps | Avg PnL: %+.2f bps", net_pnl_bps, avg_pnl_bps)
        if trades:
            gross_gains = sum(t.gross_return_bps for t in trades if t.gross_return_bps > 0)
            gross_losses = abs(sum(t.gross_return_bps for t in trades if t.gross_return_bps < 0))
            profit_factor = (gross_gains / max(1e-6, gross_losses)) if gross_losses > 0 else 999.0
            log.info("    Profit Factor: %.2f | Total Fees Paid: %.2f bps", profit_factor, sum(t.total_fee_bps for t in trades))

        log.info("--- FORWARD PREDICTION ACCURACY ---")
        for tf in ["10s", "30s", "1m", "5m", "15m"]:
            data = acc_summary.get(tf, {})
            log.info(
                "    Horizon %-3s: Overall Acc: %5.1f%% (%d smp) | Active Excursion: %5.1f%% (%d sig) | Directional Sign Acc: %5.1f%% (%d sig)",
                tf,
                data.get("overall_accuracy_pct", 0.0),
                data.get("evaluated_samples", 0),
                data.get("active_precision_pct", 0.0),
                data.get("active_signals_count", 0),
                data.get("sign_accuracy_pct", 0.0),
                data.get("sign_samples", 0),
            )
        tiers = acc_summary.get("tiers", {})
        log.info(
            "    Tier Accuracies -> Tactical(1m): %.1f%% | Meso(15m): %.1f%% | Macro(15m): %.1f%%",
            tiers.get("tactical", {}).get("accuracy_pct", 0.0),
            tiers.get("meso", {}).get("accuracy_pct", 0.0),
            tiers.get("macro", {}).get("accuracy_pct", 0.0),
        )
        log.info("=" * 80)

    def start(self) -> None:
        log.info("Bootstrapping multi-horizon ring buffer from Binance REST...")
        stats = self.ring_buffer.bootstrap_from_binance_rest(limit=100)
        log.info("Bootstrap complete across all 14 timeframes: %s", stats)

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

    def stop(self) -> None:
        if not self.running:
            return
        self.running = False
        log.info("Stopping WebSocket ingestor & REST companion poller...")
        self.ws_ingestor.stop()
        self.rest_poller.stop()
        if getattr(self.ring_buffer.model_registry, "news_daemon", None) is not None:
            self.ring_buffer.model_registry.news_daemon.stop()

        elapsed_sec = time.time() - self.start_time
        acc_summary = self.evaluator.get_summary()
        trades = self.sniper_engine.closed_trades
        n_trades = len(trades)
        wins = sum(1 for t in trades if t.net_return_bps > 0)
        win_rate = (wins / n_trades * 100.0) if n_trades > 0 else 0.0
        net_pnl_bps = sum(t.net_return_bps for t in trades)
        avg_pnl_bps = (net_pnl_bps / n_trades) if n_trades > 0 else 0.0

        # Save final metrics JSON
        final_metrics = {
            "session_duration_minutes": round(elapsed_sec / 60.0, 2),
            "bars_evaluated": self.bars_evaluated,
            "total_trades": n_trades,
            "winning_trades": wins,
            "win_rate_pct": round(win_rate, 2),
            "total_net_pnl_bps": round(net_pnl_bps, 2),
            "avg_net_pnl_bps": round(avg_pnl_bps, 2),
            "accuracy_summary": acc_summary,
            "closed_trades": [t.__dict__ for t in trades],
        }
        metrics_path = REPO_ROOT / "scratch" / "live_session_final_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(final_metrics, f, indent=2)

        log.info("=" * 80)
        log.info("SESSION CONCLUDED SUCCESSFULLY")
        log.info("Total Time: %.1f min | Total Bars: %d | Closed Trades: %d", elapsed_sec / 60.0, self.bars_evaluated, n_trades)
        log.info("Win Rate: %.1f%% | Total Net PnL: %+.2f bps", win_rate, net_pnl_bps)
        log.info("Final Metrics Saved: %s", metrics_path)
        log.info("Ledger Saved: %s", self.ledger_path)
        log.info("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Hierarchical Trading & Monitoring Session Runner")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol (default: BTCUSDT)")
    parser.add_argument("--duration", type=float, default=120.0, help="Session duration in minutes (default: 120)")
    parser.add_argument("--equity", type=float, default=10000.0, help="Starting equity USDT (default: 10000)")
    args = parser.parse_args()

    session = LiveHierarchicalTradingSession(
        symbol=args.symbol,
        duration_minutes=args.duration,
        equity_usdt=args.equity,
    )
    session.start()


if __name__ == "__main__":
    main()
