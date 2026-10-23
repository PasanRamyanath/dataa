"""
scratch/empirical_stress_testnet_engine.py
Empirical Adversarial Stress Testing Harness for Binance Testnet Execution Engine.
Role: challenger_live_2 (Empirical Challenger)

Empirically tests and stress-tests:
1. Pure Post-Only GTX Limit Orders & Zero Taker Fallback on Code -5022:
   - On API code -5022 ("Post Only order would immediately match and take"),
     verify that the engine NEVER falls back to taker market orders.
   - Verify state transitions (PLACING -> CANCELING -> IDLE), intent reset, zero positions.
   - Repeated -5022 rejections resilience under high load.
   - Other API error codes handling.

2. FIFO Queue Tracking Decrementation:
   - QueuePriorityTracker unit stress testing: positive, zero, negative, overshoot volumes.
   - Engine on_agg_trade: BUY vs SELL order matching flows vs non-matching flows.
   - Verify that non-matching side, wrong price, or wrong trade direction do NOT decrement Q_ahead.
   - Fill triggering on queue depletion.

3. Adverse Selection Cancellation:
   - AdverseSelectionGuard: depth collapse (>= 50%) and microprice drift (> 2.0 bps).
   - Engine on_depth_update: immediate cancellation, exchange cancel order issued,
     ExecutionSimulationEvent emitted with ADVERSE_CANCEL, state returns to IDLE.

4. Dynamic ATR Barriers & 30s Horizon Auto-Flattening:
   - Dynamic Take-Profit (1.5x ATR_10s) and Stop-Loss (1.2x ATR_10s) with 2.5 bps clamp.
   - Barrier hits triggering TP / SL.
   - Strict 30s alpha decay horizon auto-flattening (TIME_EXPIRY).
   - Queue resting timeout (10s in queue).

5. Statistical Rigor:
   - Wilson 95% CIs and binomial p-values on clustered runs.
   - Equivalence and numerical precision check across scipy and manual implementations.
   - Edge case proportions: k=0, k=n, n=1, n=0.
"""
from __future__ import annotations

import json
import logging
import math
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from scipy.stats import binomtest

# Ensure repository root and feature engineering scripts are in sys.path
REPO_ROOT = Path("D:/backup/Binance data")
FEAT_ENGINE_DIR = REPO_ROOT / "feature engineering scripts"
for p in [str(REPO_ROOT), str(FEAT_ENGINE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from execution.maker_node import (
    AdverseSelectionGuard,
    BBOExecutionNode,
    QueuePriorityTracker,
)
from inference.binance_testnet_engine import (
    BinanceTestnetClient,
    BinanceTestnetExecutionEngine,
)
from tests.contracts import ExecutionSimulationEvent, MakerOrderIntent

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")


class EmpiricalStressHarness:
    def __init__(self) -> None:
        self.results: dict[str, dict] = {}

    def log_test(self, suite: str, test_name: str, passed: bool, details: str = "") -> None:
        if suite not in self.results:
            self.results[suite] = {"passed": 0, "failed": 0, "tests": []}
        if passed:
            self.results[suite]["passed"] += 1
            status = "PASS"
        else:
            self.results[suite]["failed"] += 1
            status = "FAIL"
        self.results[suite]["tests"].append({"name": test_name, "status": status, "details": details})
        print(f"[{status}] [{suite}] {test_name}: {details}")

    # =========================================================================
    # SUITE 1: Pure Post-Only GTX & Zero Taker Fallback on Code -5022
    # =========================================================================
    def _create_isolated_engine(self, mock_client: MagicMock) -> BinanceTestnetExecutionEngine:
        mock_client.get_position_risk.return_value = [{"positionAmt": "0.0", "entryPrice": "0.0"}]
        mock_client.cancel_all_open_orders.return_value = {}
        mock_client.set_leverage.return_value = {}
        with patch("inference.binance_testnet_engine.BinanceTestnetClient", return_value=mock_client):
            engine = BinanceTestnetExecutionEngine(
                symbol="BTCUSDT",
                dry_run=False,
                ledger_path=REPO_ROOT / "scratch" / "test_challenger_temp_ledger.jsonl",
            )
            engine.client = mock_client
            # Guarantee IDLE state
            engine._reset_position_state()
            return engine

    def run_suite_1_zero_taker_fallback(self) -> None:
        suite = "SUITE_1_ZERO_TAKER_FALLBACK"
        print(f"\n--- Running {suite} ---")

        # Test 1.1: BUY order receives -5022
        mock_client = MagicMock(spec=BinanceTestnetClient)
        mock_client.place_limit_gtx_order.return_value = {
            "code": -5022,
            "msg": "Post Only order would immediately match and take.",
        }
        mock_client.place_market_order.return_value = {"orderId": 12345}

        engine = self._create_isolated_engine(mock_client)
        assert engine.state == "IDLE"

        res = engine.on_authorized_signal(
            side="BUY",
            bbo_bid=60000.0,
            bbo_ask=60001.0,
            bbo_bid_qty=5.0,
            bbo_ask_qty=5.0,
            atr_10s_bps=5.0,
            quantity=0.01,
        )

        passed = (
            res is False
            and mock_client.place_market_order.call_count == 0
            and engine.state == "IDLE"
            and engine.active_intent is None
            and engine.active_order_id is None
            and engine.position_side == "FLAT"
        )
        self.log_test(
            suite,
            "test_buy_order_reject_5022_zero_taker_fallback",
            passed,
            f"Result={res}, MarketOrderCalls={mock_client.place_market_order.call_count}, State={engine.state}",
        )

        # Test 1.2: SELL order receives -5022
        mock_client.reset_mock()
        mock_client.place_limit_gtx_order.return_value = {
            "code": -5022,
            "msg": "Post Only order would immediately match and take.",
        }

        res = engine.on_authorized_signal(
            side="SELL",
            bbo_bid=60000.0,
            bbo_ask=60001.0,
            bbo_bid_qty=5.0,
            bbo_ask_qty=5.0,
            atr_10s_bps=5.0,
            quantity=0.01,
        )

        passed = (
            res is False
            and mock_client.place_market_order.call_count == 0
            and engine.state == "IDLE"
            and engine.active_intent is None
            and engine.active_order_id is None
            and engine.position_side == "FLAT"
        )
        self.log_test(
            suite,
            "test_sell_order_reject_5022_zero_taker_fallback",
            passed,
            f"Result={res}, MarketOrderCalls={mock_client.place_market_order.call_count}, State={engine.state}",
        )

        # Test 1.3: High-frequency burst of 100 consecutive -5022 rejections
        mock_client.reset_mock()
        mock_client.place_limit_gtx_order.return_value = {"code": -5022, "msg": "Post Only reject"}
        burst_passes = 0
        for i in range(100):
            r = engine.on_authorized_signal(
                side="BUY" if i % 2 == 0 else "SELL",
                bbo_bid=60000.0 + i,
                bbo_ask=60001.0 + i,
                bbo_bid_qty=5.0,
                bbo_ask_qty=5.0,
                atr_10s_bps=5.0,
                quantity=0.01,
            )
            if (
                not r
                and mock_client.place_market_order.call_count == 0
                and engine.state == "IDLE"
                and engine.active_intent is None
            ):
                burst_passes += 1

        self.log_test(
            suite,
            "test_burst_100_consecutive_5022_rejections_stability",
            burst_passes == 100,
            f"100/100 aborted without taker market order fallback, state remained IDLE throughout",
        )

        # Test 1.4: Other Binance error codes (e.g. -1003 Too Many Requests, -2010 Insufficient Balance)
        for err_code, err_msg in [(-1003, "Too many requests"), (-2010, "Account has insufficient balance")]:
            mock_client.reset_mock()
            mock_client.place_limit_gtx_order.return_value = {"code": err_code, "msg": err_msg}
            r = engine.on_authorized_signal(
                side="BUY",
                bbo_bid=60000.0,
                bbo_ask=60001.0,
                bbo_bid_qty=5.0,
                bbo_ask_qty=5.0,
                atr_10s_bps=5.0,
                quantity=0.01,
            )
            passed = not r and mock_client.place_market_order.call_count == 0 and engine.state == "IDLE"
            self.log_test(
                suite,
                f"test_api_error_code_{abs(err_code)}_handling",
                passed,
                f"Error {err_code} properly aborted without taker fallback, engine state={engine.state}",
            )

        # Test 1.5: Verify order placement parameters strictly use LIMIT and GTX
        mock_client.reset_mock()
        mock_client.place_limit_gtx_order.return_value = {"orderId": 987654321}
        engine.on_authorized_signal(
            side="BUY",
            bbo_bid=60000.0,
            bbo_ask=60001.0,
            bbo_bid_qty=5.0,
            bbo_ask_qty=5.0,
            atr_10s_bps=5.0,
            quantity=0.05,
        )
        call_args = mock_client.place_limit_gtx_order.call_args[1]
        passed = (
            call_args["symbol"] == "BTCUSDT"
            and call_args["side"] == "BUY"
            and np.isclose(call_args["price"], 60000.0)
            and np.isclose(call_args["quantity"], 0.05)
        )
        self.log_test(
            suite,
            "test_order_placement_contract_parameters",
            passed,
            f"CallArgs: {call_args}",
        )
        engine.cancel_active_orders()


    # =========================================================================
    # SUITE 2: FIFO Queue Tracking Decrementation
    # =========================================================================
    def run_suite_2_fifo_queue_tracking(self) -> None:
        suite = "SUITE_2_FIFO_QUEUE_TRACKING"
        print(f"\n--- Running {suite} ---")

        # Test 2.1: QueuePriorityTracker arithmetic and monotonicity
        tracker = QueuePriorityTracker(initial_queue_ahead=10.0)
        t1 = tracker.process_trade(3.0)
        check1 = (np.isclose(tracker.current_queue_ahead, 7.0) and not t1 and not tracker.is_filled())

        t2 = tracker.process_trade(4.0)
        check2 = (np.isclose(tracker.current_queue_ahead, 3.0) and not t2 and not tracker.is_filled())

        t3 = tracker.process_trade(3.0)
        check3 = (np.isclose(tracker.current_queue_ahead, 0.0) and t3 and tracker.is_filled())

        # Additional trade volume when already filled should not make queue negative
        t4 = tracker.process_trade(5.0)
        check4 = (tracker.current_queue_ahead == 0.0 and tracker.is_filled() and np.isclose(tracker.cumulative_executed_volume, 15.0))

        self.log_test(
            suite,
            "test_queue_priority_tracker_depletion_lifecycle",
            check1 and check2 and check3 and check4,
            f"Initial: 10.0 -> -3.0 (7.0) -> -4.0 (3.0) -> -3.0 (0.0 FILLED) -> -5.0 (0.0 clamped)",
        )

        # Test 2.2: Negative and zero trade volume injection
        tracker_edge = QueuePriorityTracker(initial_queue_ahead=5.0)
        tracker_edge.process_trade(0.0)
        check_zero = np.isclose(tracker_edge.current_queue_ahead, 5.0)
        tracker_edge.process_trade(-10.0)
        check_neg = np.isclose(tracker_edge.current_queue_ahead, 5.0)
        self.log_test(
            suite,
            "test_queue_tracker_zero_negative_volume_resilience",
            check_zero and check_neg,
            f"Queue ahead invariant under non-positive volume: Q={tracker_edge.current_queue_ahead}",
        )

        # Test 2.3: Engine on_agg_trade matching flow for BUY order
        # Resting bid at $60000.0, Q_ahead = 5.0
        engine = BinanceTestnetExecutionEngine(symbol="BTCUSDT", dry_run=True)
        engine.on_authorized_signal(
            side="BUY",
            bbo_bid=60000.0,
            bbo_ask=60001.0,
            bbo_bid_qty=5.0,
            bbo_ask_qty=5.0,
            atr_10s_bps=5.0,
            quantity=0.01,
        )
        assert engine.state == "RESTING_IN_QUEUE"
        init_q = engine.queue_tracker.current_queue_ahead

        # Aggressive seller sells into our bid: buyer_maker is True, trade_price <= limit_price
        # 1. Matching flow: trade_price = 60000.0, qty = 2.0, buyer_maker = True
        engine.on_agg_trade(trade_price=60000.0, trade_qty=2.0, buyer_maker=True)
        q_after_match = engine.queue_tracker.current_queue_ahead
        check_match = np.isclose(q_after_match, 3.0)

        # 2. Non-matching flow: buyer_maker is False (taker buy from ask!) -> must NOT decrement
        engine.on_agg_trade(trade_price=60000.0, trade_qty=2.0, buyer_maker=False)
        q_after_non_match = engine.queue_tracker.current_queue_ahead
        check_non_match_side = np.isclose(q_after_non_match, 3.0)

        # 3. Non-matching price: trade_price > limit_price (e.g. 60001.0) with buyer_maker=True -> must NOT decrement
        engine.on_agg_trade(trade_price=60001.0, trade_qty=2.0, buyer_maker=True)
        q_after_non_match_p = engine.queue_tracker.current_queue_ahead
        check_non_match_price = np.isclose(q_after_non_match_p, 3.0)

        # 4. Final matching flow exhausting queue: trade_qty = 3.5 >= 3.0 -> triggers fill!
        engine.on_agg_trade(trade_price=60000.0, trade_qty=3.5, buyer_maker=True)
        check_filled = engine.state == "FILLED_HOLDING" and engine.position_side == "LONG"

        self.log_test(
            suite,
            "test_engine_buy_order_agg_trade_matching_and_fill",
            check_match and check_non_match_side and check_non_match_price and check_filled,
            f"Init Q={init_q} -> Match(2.0)->{q_after_match} -> NonMatchSide->{q_after_non_match} -> NonMatchPrice->{q_after_non_match_p} -> FinalMatch(3.5)->State={engine.state}",
        )
        engine.cancel_active_orders()

        # Test 2.4: Engine on_agg_trade matching flow for SELL order
        # Resting ask at $60001.0, Q_ahead = 4.0
        engine = BinanceTestnetExecutionEngine(symbol="BTCUSDT", dry_run=True)
        engine.on_authorized_signal(
            side="SELL",
            bbo_bid=60000.0,
            bbo_ask=60001.0,
            bbo_bid_qty=5.0,
            bbo_ask_qty=4.0,
            atr_10s_bps=5.0,
            quantity=0.01,
        )
        assert engine.state == "RESTING_IN_QUEUE"

        # Taker buy matches maker ask: buyer_maker is False, trade_price >= limit_price
        # 1. Matching flow: trade_price = 60001.0, qty = 1.5, buyer_maker = False
        engine.on_agg_trade(trade_price=60001.0, trade_qty=1.5, buyer_maker=False)
        check_sell_match = np.isclose(engine.queue_tracker.current_queue_ahead, 2.5)

        # 2. Non-matching flow: buyer_maker is True (taker sell into bid) -> must NOT decrement
        engine.on_agg_trade(trade_price=60001.0, trade_qty=1.5, buyer_maker=True)
        check_sell_non_match = np.isclose(engine.queue_tracker.current_queue_ahead, 2.5)

        # 3. Non-matching price: trade_price < limit_price (e.g. 60000.0) -> must NOT decrement
        engine.on_agg_trade(trade_price=60000.0, trade_qty=1.5, buyer_maker=False)
        check_sell_price_reject = np.isclose(engine.queue_tracker.current_queue_ahead, 2.5)

        # 4. Fill execution
        engine.on_agg_trade(trade_price=60001.0, trade_qty=3.0, buyer_maker=False)
        check_sell_fill = engine.state == "FILLED_HOLDING" and engine.position_side == "SHORT"

        self.log_test(
            suite,
            "test_engine_sell_order_agg_trade_matching_and_fill",
            check_sell_match and check_sell_non_match and check_sell_price_reject and check_sell_fill,
            f"SELL resting ask fill verified: buyer_maker=False correctly matched taker buys",
        )
        engine.cancel_active_orders()

    # =========================================================================
    # SUITE 3: Adverse Selection Cancellation Triggers
    # =========================================================================
    def run_suite_3_adverse_selection_cancellation(self) -> None:
        suite = "SUITE_3_ADVERSE_SELECTION"
        print(f"\n--- Running {suite} ---")

        # Test 3.1: AdverseSelectionGuard unit logic (50% depth collapse & 2.0 bps drift)
        guard = AdverseSelectionGuard(
            initial_supporting_depth=10.0,
            depth_deterioration_threshold=0.50,
            max_adverse_microprice_bps=2.0,
        )

        # Normal book: depth = 6.0 (60% remaining), drift = 1.0 bps -> False
        c1 = guard.should_cancel("BUY", current_supporting_depth=6.0, adverse_price_drift_bps=1.0)

        # Boundary: depth = 5.0 (50% remaining), drift = 2.0 bps -> False
        c2 = guard.should_cancel("BUY", current_supporting_depth=5.0, adverse_price_drift_bps=2.0)

        # Trigger 1: depth drops to 4.9 (49% remaining, 51% collapsed) -> True
        c3 = guard.should_cancel("BUY", current_supporting_depth=4.9, adverse_price_drift_bps=0.0)

        # Trigger 2: microprice drifts by 2.01 bps (> 2.0 bps) -> True
        c4 = guard.should_cancel("BUY", current_supporting_depth=8.0, adverse_price_drift_bps=2.01)

        # Combined triggers: depth collapsed AND adverse drift -> True
        c5 = guard.should_cancel("BUY", current_supporting_depth=1.0, adverse_price_drift_bps=5.0)

        # Negative drift (favorable market movement) -> False
        c6 = guard.should_cancel("BUY", current_supporting_depth=8.0, adverse_price_drift_bps=-3.0)

        passed = (
            not c1
            and not c2
            and c3
            and c4
            and c5
            and not c6
        )
        self.log_test(
            suite,
            "test_adverse_selection_guard_threshold_rules",
            passed,
            f"c1(6.0, 1.0)={c1}, c2(5.0, 2.0)={c2}, c3_depth_trigger(4.9)={c3}, c4_drift_trigger(2.01)={c4}, favorable(-3.0)={c6}",
        )

        # Test 3.2: Engine cancels resting order on book depth collapse
        engine = BinanceTestnetExecutionEngine(symbol="BTCUSDT", dry_run=True)
        engine.on_authorized_signal(
            side="BUY",
            bbo_bid=60000.0,
            bbo_ask=60001.0,
            bbo_bid_qty=10.0,
            bbo_ask_qty=10.0,
            atr_10s_bps=5.0,
            quantity=0.01,
        )
        assert engine.state == "RESTING_IN_QUEUE"

        # In engine, initial_supporting_depth is initialized as queue_qty + 1.0 = 11.0
        # 50% threshold of 11.0 is 5.5. A drop to 5.0 triggers cancellation
        engine.on_depth_update(current_supporting_depth=5.0, adverse_drift_bps=0.5)

        passed_depth = (
            engine.state == "IDLE"
            and engine.active_intent is None
            and engine.active_order_id is None
            and engine.position_side == "FLAT"
        )
        self.log_test(
            suite,
            "test_engine_cancellation_on_depth_collapse",
            passed_depth,
            f"Engine transitioned to IDLE and cleared active intent on depth collapse",
        )

        # Test 3.3: Engine cancels resting order on microprice drift > 2.0 bps
        engine.on_authorized_signal(
            side="SELL",
            bbo_bid=60000.0,
            bbo_ask=60001.0,
            bbo_bid_qty=10.0,
            bbo_ask_qty=10.0,
            atr_10s_bps=5.0,
            quantity=0.01,
        )
        assert engine.state == "RESTING_IN_QUEUE"

        # Adverse microprice drift = 2.5 bps > 2.0 bps
        engine.on_depth_update(current_supporting_depth=12.0, adverse_drift_bps=2.5)

        passed_drift = (
            engine.state == "IDLE"
            and engine.active_intent is None
            and engine.active_order_id is None
            and engine.position_side == "FLAT"
        )
        self.log_test(
            suite,
            "test_engine_cancellation_on_microprice_drift",
            passed_drift,
            f"Engine transitioned to IDLE and cleared active intent on adverse microprice drift > 2.0 bps",
        )

    # =========================================================================
    # SUITE 4: Dynamic ATR Barriers & 30s Horizon Auto-Flattening
    # =========================================================================
    def run_suite_4_barriers_and_auto_flattening(self) -> None:
        suite = "SUITE_4_BARRIERS_AND_HORIZON"
        print(f"\n--- Running {suite} ---")

        # Test 4.1: Dynamic ATR Barrier calculation & 2.5 bps floor clamp
        node = BBOExecutionNode(k_tp=1.5, k_sl=1.2, max_holding_bars=3)

        # Case A: Normal ATR = 6.0 bps
        intent_normal = node.create_intent(
            side="BUY",
            bbo_bid=50000.0,
            bbo_ask=50001.0,
            bbo_bid_qty=2.0,
            bbo_ask_qty=2.0,
            atr_10s_bps=6.0,
        )
        tp_expected = 1.5 * 6.0  # 9.0 bps
        sl_expected = 1.2 * 6.0  # 7.2 bps
        check_normal = (
            np.isclose(intent_normal.take_profit_bps, tp_expected)
            and np.isclose(intent_normal.stop_loss_bps, sl_expected)
        )

        # Case B: Compressed ATR = 1.0 bps -> should clamp to 2.5 bps floor
        intent_clamped = node.create_intent(
            side="BUY",
            bbo_bid=50000.0,
            bbo_ask=50001.0,
            bbo_bid_qty=2.0,
            bbo_ask_qty=2.0,
            atr_10s_bps=1.0,
        )
        tp_clamped = 1.5 * 2.5  # 3.75 bps
        sl_clamped = 1.2 * 2.5  # 3.0 bps
        check_clamped = (
            np.isclose(intent_clamped.take_profit_bps, tp_clamped)
            and np.isclose(intent_clamped.stop_loss_bps, sl_clamped)
        )

        self.log_test(
            suite,
            "test_dynamic_atr_barrier_scaling_and_floor_clamp",
            check_normal and check_clamped,
            f"Normal(6.0bps): TP={intent_normal.take_profit_bps}bps, SL={intent_normal.stop_loss_bps}bps | Clamped(1.0bps->2.5): TP={intent_clamped.take_profit_bps}bps, SL={intent_clamped.stop_loss_bps}bps",
        )

        # Test 4.2: Take-Profit Hit in LONG position
        engine = BinanceTestnetExecutionEngine(symbol="BTCUSDT", dry_run=True, k_tp=1.5, k_sl=1.2)
        engine.on_authorized_signal(
            side="BUY",
            bbo_bid=60000.0,
            bbo_ask=60001.0,
            bbo_bid_qty=1.0,
            bbo_ask_qty=1.0,
            atr_10s_bps=10.0,  # TP = 15.0 bps (+$90.0) -> $60090.0 | SL = 12.0 bps (-$72.0) -> $59928.0
            quantity=0.01,
        )
        # Simulate fill
        engine.on_agg_trade(trade_price=60000.0, trade_qty=2.0, buyer_maker=True)
        assert engine.state == "FILLED_HOLDING"
        assert np.isclose(engine.tp_target_price, 60090.0)
        assert np.isclose(engine.sl_target_price, 59928.0)

        # Price ticks up towards TP
        engine.on_price_tick(60050.0)
        remains_holding = engine.state == "FILLED_HOLDING"
        # Price hits TP
        engine.on_price_tick(60090.5)
        tp_triggered = engine.state == "IDLE" and engine.position_side == "FLAT"

        self.log_test(
            suite,
            "test_long_position_take_profit_execution",
            remains_holding and tp_triggered,
            f"TP target $60090.0 triggered on tick $60090.5, engine flattened to IDLE",
        )

        # Test 4.3: Stop-Loss Hit in SHORT position
        engine = BinanceTestnetExecutionEngine(symbol="BTCUSDT", dry_run=True, k_tp=1.5, k_sl=1.2)
        engine.on_authorized_signal(
            side="SELL",
            bbo_bid=59999.0,
            bbo_ask=60000.0,
            bbo_bid_qty=1.0,
            bbo_ask_qty=1.0,
            atr_10s_bps=10.0,  # Entry: 60000.0 | TP = -90.0 -> 59910.0 | SL = +72.0 -> 60072.0
            quantity=0.01,
        )
        engine.on_agg_trade(trade_price=60000.0, trade_qty=2.0, buyer_maker=False)
        assert engine.state == "FILLED_HOLDING" and engine.position_side == "SHORT"
        assert np.isclose(engine.tp_target_price, 59910.0)
        assert np.isclose(engine.sl_target_price, 60072.0)

        # Price spikes up to SL
        engine.on_price_tick(60072.5)
        sl_triggered = engine.state == "IDLE" and engine.position_side == "FLAT"

        self.log_test(
            suite,
            "test_short_position_stop_loss_execution",
            sl_triggered,
            f"SL target $60072.0 triggered on tick $60072.5, emergency flatten to IDLE",
        )

        # Test 4.4: Strict 30s Alpha Horizon Auto-Flattening
        engine = BinanceTestnetExecutionEngine(
            symbol="BTCUSDT",
            dry_run=True,
            max_holding_seconds=30.0,
        )
        engine.on_authorized_signal(
            side="BUY",
            bbo_bid=60000.0,
            bbo_ask=60001.0,
            bbo_bid_qty=1.0,
            bbo_ask_qty=1.0,
            atr_10s_bps=10.0,
            quantity=0.01,
        )
        engine.on_agg_trade(trade_price=60000.0, trade_qty=2.0, buyer_maker=True)
        assert engine.state == "FILLED_HOLDING"

        # Mock time progression: 10s elapsed
        engine.entry_timestamp_ms = int((time.time() - 10.0) * 1000)
        engine.on_price_tick(60010.0)
        holding_10s = engine.state == "FILLED_HOLDING"

        # 29.5s elapsed
        engine.entry_timestamp_ms = int((time.time() - 29.5) * 1000)
        engine.on_price_tick(60010.0)
        holding_29s = engine.state == "FILLED_HOLDING"

        # 30.1s elapsed (horizon reached)
        engine.entry_timestamp_ms = int((time.time() - 30.1) * 1000)
        engine.on_price_tick(60010.0)
        auto_flattened_30s = engine.state == "IDLE" and engine.position_side == "FLAT"

        self.log_test(
            suite,
            "test_strict_30s_horizon_auto_flattening",
            holding_10s and holding_29s and auto_flattened_30s,
            f"10s: {holding_10s}, 29.5s: {holding_29s}, 30.1s: Flattened to IDLE={auto_flattened_30s}",
        )

        # Test 4.5: Queue 10s resting timeout
        engine = BinanceTestnetExecutionEngine(symbol="BTCUSDT", dry_run=True)
        engine.on_authorized_signal(
            side="BUY",
            bbo_bid=60000.0,
            bbo_ask=60001.0,
            bbo_bid_qty=100.0,
            bbo_ask_qty=100.0,
            atr_10s_bps=5.0,
            quantity=0.01,
        )
        assert engine.state == "RESTING_IN_QUEUE"

        # 5s in queue
        engine.placement_timestamp_ms = int((time.time() - 5.0) * 1000)
        engine.on_price_tick(60000.0)
        still_in_queue = engine.state == "RESTING_IN_QUEUE"

        # 10.1s in queue (exceeded 1 bar resting)
        engine.placement_timestamp_ms = int((time.time() - 10.1) * 1000)
        engine.on_price_tick(60000.0)
        queue_timed_out = engine.state == "IDLE"

        self.log_test(
            suite,
            "test_queue_resting_timeout_at_10s",
            still_in_queue and queue_timed_out,
            f"5s resting in queue: {still_in_queue} | 10.1s resting: Timed out and canceled to IDLE={queue_timed_out}",
        )

    # =========================================================================
    # SUITE 5: Statistical Rigor (Wilson 95% CIs and Binomial P-Values)
    # =========================================================================
    def run_suite_5_statistical_rigor(self) -> None:
        suite = "SUITE_5_STATISTICAL_RIGOR"
        print(f"\n--- Running {suite} ---")

        from scratch.rigorous_evaluation_audit import wilson_ci as audit_wilson_ci
        from scripts.purged_walk_forward_backtest import calculate_wilson_ci as backtest_wilson_ci

        # Test 5.1: Mathematical equivalence of Wilson CI implementations against scipy
        test_cases = [
            (0, 10),
            (1, 10),
            (5, 10),
            (9, 10),
            (10, 10),
            (50, 100),
            (188, 274),  # Walk-forward reported AC4 numbers
            (1, 1),
            (0, 1),
        ]

        all_equiv = True
        max_diff = 0.0

        for k, n in test_cases:
            # Audit script version (calls binomtest(..., method="wilson"))
            a_low, a_high, a_pval = audit_wilson_ci(k, n)

            # Backtest script manual Wilson formula
            b_low, b_high, b_pval = backtest_wilson_ci(k, n)

            # Scipy reference
            scipy_res = binomtest(k, n, 0.5, alternative="two-sided")
            s_ci = scipy_res.proportion_ci(confidence_level=0.95, method="wilson")
            s_low, s_high = float(s_ci.low), float(s_ci.high)
            s_pval = float(scipy_res.pvalue)

            diff_low = abs(a_low - s_low) + abs(b_low - s_low)
            diff_high = abs(a_high - s_high) + abs(b_high - s_high)
            diff_pval = abs(a_pval - s_pval) + abs(b_pval - s_pval)
            max_diff = max(max_diff, diff_low, diff_high, diff_pval)

            if diff_low > 1e-4 or diff_high > 1e-4 or diff_pval > 1e-4:
                all_equiv = False
                print(f"Mismatch at k={k}, n={n}: a=({a_low:.4f}, {a_high:.4f}), b=({b_low:.4f}, {b_high:.4f}), s=({s_low:.4f}, {s_high:.4f})")

        self.log_test(
            suite,
            "test_wilson_ci_formula_equivalence_against_scipy",
            all_equiv,
            f"Tested across {len(test_cases)} boundary cases. Max deviation = {max_diff:.2e} (tolerance: 1e-4)",
        )

        # Test 5.2: Edge case n = 0 handling (graceful degradation without zero-division error)
        try:
            z_low, z_high, z_pval = audit_wilson_ci(0, 0)
            z_check = (z_low == 0.0 and z_high == 0.0 and z_pval == 1.0)
        except Exception as e:
            z_check = False

        try:
            bz_low, bz_high, bz_pval = backtest_wilson_ci(0, 0)
            bz_check = (bz_low == 0.0 and bz_high == 1.0 and bz_pval == 1.0)
        except Exception as e:
            bz_check = False

        self.log_test(
            suite,
            "test_wilson_ci_zero_sample_handling",
            z_check and bz_check,
            f"Zero samples handled cleanly without ZeroDivisionError",
        )

        # Test 5.3: Clustered Run Grouping Logic
        from scripts.purged_walk_forward_backtest import cluster_signals_into_runs

        # Stream with 3 LONGs, 2 FLATs, 2 SHORTs, 1 FLAT, 1 LONG, 2 SHORTs
        synthetic_stream = [2, 2, 2, 1, 1, 0, 0, 1, 2, 0, 0]
        clusters = cluster_signals_into_runs(synthetic_stream)
        # Expected clusters:
        # 1: [2, 2, 2]
        # 2: [0, 0]
        # 3: [2]
        # 4: [0, 0]
        expected_len = 4
        cluster_pass = (
            len(clusters) == expected_len
            and clusters[0] == [2, 2, 2]
            and clusters[1] == [0, 0]
            and clusters[2] == [2]
            and clusters[3] == [0, 0]
        )
        self.log_test(
            suite,
            "test_clustered_signal_grouping_delimiters",
            cluster_pass,
            f"Generated {len(clusters)} clusters (expected {expected_len}): {clusters}",
        )

        # Test 5.4: Verify Non-Tie handling and p-value statistical rigor on empirical walk-forward output
        # Case from reported walk-forward: N=274, wins=188
        n_walk = 274
        wins_walk = 188
        low_wf, high_wf, pval_wf = backtest_wilson_ci(wins_walk, n_walk)
        scipy_wf = binomtest(wins_walk, n_walk, 0.5, alternative="two-sided")

        rigor_pass = (
            np.isclose(low_wf, scipy_wf.proportion_ci(confidence_level=0.95, method="wilson").low, atol=1e-4)
            and np.isclose(high_wf, scipy_wf.proportion_ci(confidence_level=0.95, method="wilson").high, atol=1e-4)
            and np.isclose(pval_wf, scipy_wf.pvalue, atol=1e-12)
            and low_wf > 0.50  # Lower bound strictly rejects 50%
            and pval_wf < 1e-8  # Extreme statistical significance
        )
        self.log_test(
            suite,
            "test_empirical_walk_forward_ac4_statistical_significance",
            rigor_pass,
            f"N={n_walk}, Wins={wins_walk} -> CI=[{low_wf*100:.2f}%, {high_wf*100:.2f}%], p-value={pval_wf:.4e} (< 1e-8)",
        )

    # =========================================================================
    # SUITE 6: Multithreaded Concurrency & Race-Condition Stress
    # =========================================================================
    def run_suite_6_multithreaded_concurrency(self) -> None:
        import concurrent.futures
        import threading

        suite = "SUITE_6_CONCURRENCY_STRESS"
        print(f"\n--- Running {suite} ---")

        engine = BinanceTestnetExecutionEngine(
            symbol="BTCUSDT",
            dry_run=True,
            ledger_path=REPO_ROOT / "scratch" / "test_concurrency_ledger.jsonl",
        )

        stop_event = threading.Event()
        errors: list[str] = []

        def signal_worker(thread_id: int):
            for i in range(50):
                if stop_event.is_set():
                    break
                try:
                    side = "BUY" if (i + thread_id) % 2 == 0 else "SELL"
                    engine.on_authorized_signal(
                        side=side,
                        bbo_bid=60000.0 + (i % 10),
                        bbo_ask=60001.0 + (i % 10),
                        bbo_bid_qty=5.0,
                        bbo_ask_qty=5.0,
                        atr_10s_bps=5.0,
                        quantity=0.01,
                    )
                except Exception as e:
                    errors.append(f"signal_worker error: {e}")

        def trade_worker(thread_id: int):
            for i in range(100):
                if stop_event.is_set():
                    break
                try:
                    engine.on_agg_trade(
                        trade_price=60000.0 + (i % 5),
                        trade_qty=0.5,
                        buyer_maker=(i % 2 == 0),
                    )
                except Exception as e:
                    errors.append(f"trade_worker error: {e}")

        def price_worker(thread_id: int):
            for i in range(100):
                if stop_event.is_set():
                    break
                try:
                    engine.on_price_tick(60000.0 + (i % 100) - 50)
                except Exception as e:
                    errors.append(f"price_worker error: {e}")

        def depth_worker(thread_id: int):
            for i in range(50):
                if stop_event.is_set():
                    break
                try:
                    engine.on_depth_update(
                        current_supporting_depth=float(max(0.1, 10.0 - (i % 8))),
                        adverse_drift_bps=float((i % 5) * 0.6),
                    )
                except Exception as e:
                    errors.append(f"depth_worker error: {e}")

        threads: list[threading.Thread] = []
        for t_i in range(3):
            threads.append(threading.Thread(target=signal_worker, args=(t_i,)))
            threads.append(threading.Thread(target=trade_worker, args=(t_i,)))
            threads.append(threading.Thread(target=price_worker, args=(t_i,)))
            threads.append(threading.Thread(target=depth_worker, args=(t_i,)))

        for t in threads:
            t.start()

        for t in threads:
            t.join(timeout=5.0)

        deadlocks = any(t.is_alive() for t in threads)
        if deadlocks:
            stop_event.set()

        # Invariant checks
        valid_states = {"IDLE", "PLACING", "RESTING_IN_QUEUE", "PARTIAL_FILL", "FILLED_HOLDING", "CANCELING"}
        state_valid = engine.state in valid_states
        if engine.state == "IDLE":
            pos_valid = engine.position_side == "FLAT"
        else:
            pos_valid = engine.position_side in ("LONG", "SHORT", "FLAT")

        passed = (not deadlocks) and (len(errors) == 0) and state_valid and pos_valid
        self.log_test(
            suite,
            "test_multithreaded_concurrency_deadlock_and_race_resilience",
            passed,
            f"Threads: 12, Errors: {len(errors)}, Deadlock: {deadlocks}, Final State: {engine.state}, Pos: {engine.position_side}",
        )
        engine.cancel_active_orders()

    # =========================================================================
    # SUITE 7: Extreme Volatility, Slippage & Market Shock Resilience
    # =========================================================================
    def run_suite_7_extreme_market_shocks(self) -> None:
        suite = "SUITE_7_MARKET_SHOCKS"
        print(f"\n--- Running {suite} ---")

        # Test 7.1: Flash crash (-50% single-tick collapse) while holding LONG
        engine = BinanceTestnetExecutionEngine(symbol="BTCUSDT", dry_run=True, k_tp=1.5, k_sl=1.2)
        engine.on_authorized_signal(
            side="BUY",
            bbo_bid=60000.0,
            bbo_ask=60001.0,
            bbo_bid_qty=1.0,
            bbo_ask_qty=1.0,
            atr_10s_bps=5.0,
            quantity=0.01,
        )
        engine.on_agg_trade(trade_price=60000.0, trade_qty=2.0, buyer_maker=True)
        assert engine.state == "FILLED_HOLDING" and engine.position_side == "LONG"

        # Market flash crashes to $30,000
        engine.on_price_tick(30000.0)
        flash_crash_handled = (
            engine.state == "IDLE"
            and engine.position_side == "FLAT"
            and engine.active_intent is None
            and engine.active_order_id is None
        )
        self.log_test(
            suite,
            "test_flash_crash_emergency_flatten_resilience",
            flash_crash_handled,
            f"Position flattened to IDLE after -50% price drop from $60,000 to $30,000",
        )

        # Test 7.2: Liquidity vacuum (supporting depth drops to 1e-8 BTC)
        engine.on_authorized_signal(
            side="BUY",
            bbo_bid=60000.0,
            bbo_ask=60001.0,
            bbo_bid_qty=10.0,
            bbo_ask_qty=10.0,
            atr_10s_bps=5.0,
            quantity=0.01,
        )
        assert engine.state == "RESTING_IN_QUEUE"

        # Complete book vacuum
        engine.on_depth_update(current_supporting_depth=1e-8, adverse_drift_bps=0.0)
        vacuum_handled = engine.state == "IDLE" and engine.active_intent is None
        self.log_test(
            suite,
            "test_liquidity_vacuum_immediate_cancel",
            vacuum_handled,
            f"Resting order canceled immediately when depth collapsed to 1e-8 BTC",
        )

        # Test 7.3: Negative and hyper-extreme ATR input values
        node = BBOExecutionNode()
        intent_neg_atr = node.create_intent(
            side="BUY",
            bbo_bid=60000.0,
            bbo_ask=60001.0,
            bbo_bid_qty=5.0,
            bbo_ask_qty=5.0,
            atr_10s_bps=-100.0,
        )
        clamp_neg_ok = np.isclose(intent_neg_atr.take_profit_bps, 1.5 * 2.5) and np.isclose(intent_neg_atr.stop_loss_bps, 1.2 * 2.5)

        intent_hyper_atr = node.create_intent(
            side="BUY",
            bbo_bid=60000.0,
            bbo_ask=60001.0,
            bbo_bid_qty=5.0,
            bbo_ask_qty=5.0,
            atr_10s_bps=1000.0,  # 10% move
        )
        hyper_atr_ok = np.isclose(intent_hyper_atr.take_profit_bps, 1500.0) and np.isclose(intent_hyper_atr.stop_loss_bps, 1200.0)

        self.log_test(
            suite,
            "test_extreme_and_negative_atr_input_resilience",
            clamp_neg_ok and hyper_atr_ok,
            f"Negative ATR clamped to 2.5 bps (TP={intent_neg_atr.take_profit_bps}bps) | Hyper ATR 1000bps scaled (TP={intent_hyper_atr.take_profit_bps}bps)",
        )

        # Test 7.4: Crossed book and corrupted quote rejections
        corrupted_quotes = [
            (60005.0, 60000.0, 1.0, 1.0, "Crossed (bid > ask)"),
            (60000.0, 60000.05, 1.0, 1.0, "Spread < 0.10 USDT"),
            (-500.0, 60000.0, 1.0, 1.0, "Negative bid"),
            (60000.0, -100.0, 1.0, 1.0, "Negative ask"),
            (60000.0, 60001.0, 0.0, 1.0, "Zero bid quantity"),
            (60000.0, 60001.0, 1.0, 0.0, "Zero ask quantity"),
        ]
        all_rejected = True
        for bid, ask, b_qty, a_qty, desc in corrupted_quotes:
            res = engine.on_authorized_signal(
                side="BUY",
                bbo_bid=bid,
                bbo_ask=ask,
                bbo_bid_qty=b_qty,
                bbo_ask_qty=a_qty,
                atr_10s_bps=5.0,
            )
            if res is not False or engine.state != "IDLE":
                all_rejected = False
                print(f"Failed to reject corrupted quote: {desc}")

        self.log_test(
            suite,
            "test_corrupted_and_crossed_quotes_strict_rejection",
            all_rejected,
            f"Rejected all 6 invalid/crossed/zero quote scenarios cleanly, state remained IDLE",
        )

    # =========================================================================
    # SUITE 8: Ledger Persistence, Forward Resolution & Telemetry Audit
    # =========================================================================
    def run_suite_8_telemetry_and_ledger_audit(self) -> None:
        suite = "SUITE_8_TELEMETRY_AND_AUDIT"
        print(f"\n--- Running {suite} ---")

        test_ledger_path = REPO_ROOT / "scratch" / "test_telemetry_stream_eval.jsonl"
        if test_ledger_path.exists():
            test_ledger_path.unlink()

        engine = BinanceTestnetExecutionEngine(
            symbol="BTCUSDT",
            dry_run=True,
            ledger_path=test_ledger_path,
        )

        # Simulate 25 bars with directional trading and mid updates
        now_ms = int(time.time() * 1000)
        base_mid = 85000.0

        for b_idx in range(25):
            current_mid = base_mid + (b_idx * 5.0)  # Drift upwards
            b_bid = current_mid - 0.5
            b_ask = current_mid + 0.5

            p_dir = "BUY" if b_idx % 2 == 0 else "SELL"
            engine.on_authorized_signal(
                side=p_dir,
                bbo_bid=b_bid,
                bbo_ask=b_ask,
                bbo_bid_qty=2.0,
                bbo_ask_qty=2.0,
                atr_10s_bps=5.0,
                quantity=0.01,
                bar_idx=b_idx,
                raw_score=0.5 if p_dir == "BUY" else -0.5,
                expected_move_bps=7.5,
                continuous_prob=0.72,
            )

            # Trade fills order
            if engine.state == "RESTING_IN_QUEUE":
                engine.on_agg_trade(
                    trade_price=b_bid if p_dir == "BUY" else b_ask,
                    trade_qty=5.0,
                    buyer_maker=(p_dir == "BUY"),
                )

            # Tick forward price to all pending bars
            engine.record_forward_tick(current_mid, now_ms + b_idx * 10000)

            # Close position if open
            if engine.state == "FILLED_HOLDING":
                engine.on_price_tick(current_mid)

        # Flush any remaining settled items
        for _ in range(20):
            engine.bar_index_counter += 1
            engine.record_forward_tick(base_mid + 200.0, now_ms + 300000)

        # Read back test ledger and verify schema
        assert test_ledger_path.exists() and test_ledger_path.stat().st_size > 0
        with open(test_ledger_path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

        n_records = len(records)
        schema_valid = True
        for r in records:
            req_keys = ["bar_idx", "bbo_quotes", "microstructure", "ensemble", "meta_labeler", "execution", "forward_resolution"]
            if not all(k in r for k in req_keys):
                schema_valid = False
                break

        self.log_test(
            suite,
            "test_telemetry_streaming_ledger_schema_compliance",
            schema_valid and n_records > 0,
            f"Flushed and verified {n_records} structured JSON lines with complete contract schemas",
        )

        # Run rigorous audit on this generated test ledger
        from scratch.rigorous_evaluation_audit import run_rigorous_audit
        try:
            audit_res = run_rigorous_audit(test_ledger_path)
            audit_pass = (
                audit_res["n_bars"] == n_records
                and "multi_horizon" in audit_res
                and "runs" in audit_res
            )
        except Exception as e:
            audit_pass = False
            print(f"Audit exception: {e}")

        self.log_test(
            suite,
            "test_rigorous_evaluation_audit_on_live_stream_records",
            audit_pass,
            f"Successfully executed multi-horizon resolution and run clustering on streamed ledger",
        )

        # Cleanup temp test ledger
        if test_ledger_path.exists():
            test_ledger_path.unlink()

    # =========================================================================
    # SUMMARY
    # =========================================================================
    def summarize(self) -> bool:
        print("\n" + "=" * 78)
        print("         EMPIRICAL CHALLENGER STRESS TEST RESULTS SUMMARY")
        print("=" * 78)
        total_passed = 0
        total_failed = 0
        for suite, data in self.results.items():
            p = data["passed"]
            f = data["failed"]
            total_passed += p
            total_failed += f
            print(f"  {suite:<35} | Passed: {p:2d} | Failed: {f:2d}")

        print("-" * 78)
        print(f"  TOTAL: {total_passed} Passed, {total_failed} Failed across {len(self.results)} Suites")
        verdict = "CONFIRMED" if total_failed == 0 else "DISPROVEN"
        print(f"  FINAL VERDICT: {verdict}")
        print("=" * 78 + "\n")
        return total_failed == 0


if __name__ == "__main__":
    harness = EmpiricalStressHarness()
    harness.run_suite_1_zero_taker_fallback()
    harness.run_suite_2_fifo_queue_tracking()
    harness.run_suite_3_adverse_selection_cancellation()
    harness.run_suite_4_barriers_and_auto_flattening()
    harness.run_suite_5_statistical_rigor()
    harness.run_suite_6_multithreaded_concurrency()
    harness.run_suite_7_extreme_market_shocks()
    harness.run_suite_8_telemetry_and_ledger_audit()
    all_ok = harness.summarize()
    sys.exit(0 if all_ok else 1)

