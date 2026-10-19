# Handoff Report: Autonomous BBO Maker Execution on Binance USD-M Testnet

**Agent:** `explorer_r3_testnet`  
**Working Directory:** `D:\backup\Binance data\.agents\teamwork\explorer_r3_testnet`  
**Handoff Type:** Hard (Task Complete)  
**Parent Orchestrator:** `6f56ce3c-0c13-49cc-a255-279a76a73637`  
**Primary Deliverable:** [`testnet_execution_strategy.md`](file:///D:/backup/Binance%20data/.agents/teamwork/explorer_r3_testnet/testnet_execution_strategy.md)  
**Date:** 2026-10-18  

---

## 1. Observation

1. **Test Suite Status:**
   - Ran command: `pytest tests/test_leakage_and_features.py -v`
   - Result: 48 passed in 2.28s.
   - Specific tests validating execution contracts:
     - `test_gtx_maker_order_intent_contract`: Validates `MakerOrderIntent` requires `time_in_force == "GTX"` and `max_holding_bars == 3`.
     - `test_maker_order_rejects_non_gtx_or_invalid_horizon`: Confirms rejection of `GTC` or holding horizons $> 3$ bars (`tests/test_leakage_and_features.py:488-519`).
     - `test_queue_priority_and_fill_tracking`: Verifies FIFO queue depletion by aggressive trades (`tests/test_leakage_and_features.py:520-532`).
     - `test_adverse_selection_guard_triggers_cancel`: Confirms cancellation when supporting depth collapses by $\ge 50\%$ (`tests/test_leakage_and_features.py:533-541`).
     - `test_alpha_horizon_sync_and_dynamic_atr_barriers`: Confirms 30s horizon and dynamic ATR-scaled barriers with VIP0 maker fee ($4.0\text{ bps}$ RT) (`tests/test_leakage_and_features.py:542-566`).

2. **Analysis of `feature engineering scripts/execution/maker_node.py`:**
   - Implements `QueuePriorityTracker` (lines 31–61) with `process_trade(trade_volume)` and `is_filled()`.
   - Implements `AdverseSelectionGuard` (lines 63–98) with `should_cancel(side, current_supporting_depth, adverse_price_drift_bps)`.
   - Implements `BBOExecutionNode` (lines 100–317) with `create_intent()` and `simulate_trade()`.
   - Observation: `BBOExecutionNode` is currently built for historical backtesting and purged walk-forward validation; it does not contain live REST order submission or real-time WebSocket state management.

3. **Forensic Analysis of Legacy Compiled `binance_testnet_engine.pyc`:**
   - Decompiled constants and bytecode from `feature engineering scripts/inference/__pycache__/binance_testnet_engine.cpython-313.pyc`:
     - Discovered `BinanceTestnetClient` (`sync_server_time`, `send_signed_request`) and `BinanceTestnetTradingEngine` (`_init_exchange`, `sync_position_from_exchange`, `open_position`, `close_position`, `on_tick`, `get_telemetry`, `_log_trade_record`).
     - Constant string in `open_position`: `">>> [BINANCE TESTNET] GTX Post-Only would cross spread (-5022). Smart Hybrid routing MARKET order..."`.
     - Constant string in `close_position`: `"type": "MARKET", "reduceOnly": "true"`.
     - Static barriers in `__init__`: `tp_pct=0.0016` (+16 bps), `sl_pct=0.0012` (-12 bps), `max_holding_bars=18` (180s / 3 min).

4. **Binance Futures Testnet API Mechanics in Scratch Scripts:**
   - `feature engineering scripts/scratch/test_binance_testnet_order.py`:
     - Base URL: `https://testnet.binancefuture.com`
     - Uses `timeInForce: "GTX"` (Post-Only Maker) at line 70.
     - Places and cancels limit order successfully on Binance USD-M Testnet.
   - `feature engineering scripts/scratch/test_binance_testnet_auth.py`:
     - Tested both endpoints: `https://testnet.binancefuture.com` and `https://demo-fapi.binance.com`.
     - Confirmed HMAC-SHA256 signature against `/fapi/v2/account`.
   - `feature engineering scripts/scratch/test_binance_testnet_account.py`:
     - Verified `/fapi/v2/positionRisk`, leverage setting `/fapi/v1/leverage`, and `/fapi/v1/openOrders`.

5. **Empirical Ledger Audit Findings (`ORIGINAL_REQUEST.md:15-20` and `survey_exec_report.md`):**
   - Alpha peaks at 20–30s ($H3$) with mean move $\approx 0.74\text{ bps}$.
   - Holding trades for 180s caused 100% time-stop expirations against secular drift (+25.18 bps upward trend over 45 min).
   - Taker fees are 10.0 bps RT, which mathematically guarantees negative net PnL ($-9.26\text{ bps}$ per trade). VIP0 maker fees are 4.0 bps RT (2.0 bps entry + 2.0 bps exit).

---

## 2. Logic Chain

1. **Step 1 (Why Taker Fallback Must Be Eliminated):**
   - From Observation 5, peak alpha at 30s is $+0.74\text{ bps}$.
   - From Observation 3, the legacy testnet engine fell back to a `MARKET` taker order whenever Binance returned `-5022` (`"Post Only order would immediately match and take"`).
   - Taker fee on Binance USD-M is $0.05\%$ per leg ($10.0\text{ bps}$ RT).
   - Therefore, any execution path routing market taker orders destroys $13\times$ the alpha edge. Pure maker execution must strictly reject or reprice on code `-5022` with zero taker fallback.

2. **Step 2 (Why 180s and Static 16/12 bps Barriers Failed):**
   - From Observation 3 and 5, static barriers of $+16.0\text{ bps}$ and $-12.0\text{ bps}$ over a 180s window were chosen based on legacy sequence models.
   - Over 30s (3 bars), BTC 1-standard-deviation move is $\approx 1.28\text{ bps}$.
   - A 16 bps barrier is $> 12\sigma$ away, resulting in a 0.0% TP hit rate and 100% time-stop expirations into secular market drift.
   - Therefore, dynamic ATR barriers ($k_{\text{tp}} = 1.5 \times \text{ATR}_{10s} \approx 3.0\text{–}6.0\text{ bps}$, $k_{\text{sl}} = 1.2 \times \text{ATR}_{10s} \approx 2.4\text{–}4.8\text{ bps}$) with a strict 30.0s time limit are mathematically required to harvest the peak alpha.

3. **Step 3 (Why FIFO Queue Tracking and Adverse Selection Guard are Mandatory):**
   - In a pure maker model, resting at the BBO does not guarantee a fill; the order sits behind resting volume $Q_{\text{ahead}}$.
   - If informed aggressive flow sweeps the book, the order fills right before a crash (adverse selection).
   - By wiring `@aggTrade` into `QueuePriorityTracker`, the engine monitors real volume consuming $Q_{\text{ahead}}$.
   - By wiring `@depth20@100ms` and `@bookTicker` into `AdverseSelectionGuard`, any collapse in supporting depth ($< 50\%$) or microprice drift ($> 2.0\text{ bps}$) triggers `DELETE /fapi/v1/order` before an adverse fill happens.

4. **Step 4 (Why a Production Testnet Engine Must Be Created):**
   - `BBOExecutionNode` in `maker_node.py` provides the quantitative logic, but cannot interact with live sockets or REST APIs.
   - The compiled legacy `binance_testnet_engine.pyc` is fatally flawed and has no source `.py` file present in `feature engineering scripts/inference/`.
   - Therefore, a clean, modular `feature engineering scripts/inference/binance_testnet_engine.py` must be implemented, following the 7-section technical strategy formulated in `testnet_execution_strategy.md`.

---

## 3. Caveats

1. **Testnet Liquidity Disparities:** Binance USD-M Testnet (`testnet.binancefuture.com`) has shallower liquidity and occasional artificial tick gaps compared to live production (`fstream.binance.com`). The engine incorporates safety fallbacks (checking BBO validity and clamping ATR to $\ge 2.5\text{ bps}$) to prevent spurious orders during testnet illiquidity.
2. **Clock Drift on Windows:** Local Windows system clocks can drift by $\pm 500\text{ms}$ or more, which triggers Binance HTTP 400 (`-1021 INVALID_TIMESTAMP`). The strategy implements mandatory server time synchronization via `GET /fapi/v1/time` on startup and every 30 minutes.
3. **Emergency Stop-Loss Fee Tier:** When an emergency stop-loss or 30-second time-expiry triggers, flattening exposure requires an immediate reduce-only order. If routed as aggressive market order, that single leg pays taker fee (5.0 bps), making total RT fee 7.0 bps. The model accounts for this in net PnL calculations.

---

## 4. Conclusion

1. **Technical Strategy Completed:** An exhaustive technical strategy has been formulated and documented in [`testnet_execution_strategy.md`](file:///D:/backup/Binance%20data/.agents/teamwork/explorer_r3_testnet/testnet_execution_strategy.md).
2. **Definitive Architecture Established:**
   - **Order Type:** Pure `GTX` Post-Only Limit orders at BBO (Best Bid for BUY, Best Ask for SELL). Zero taker fallback on code `-5022`.
   - **Queue Tracking:** Real-time FIFO queue tracking using `QueuePriorityTracker` decremented by aggressive `@aggTrade` flow.
   - **Adverse Selection Protection:** Active cancellation via `AdverseSelectionGuard` when supporting depth drops by $\ge 50\%$ or microprice drifts by $> 2.0\text{ bps}$.
   - **Horizon & Dynamic Barriers:** Strict 3-bar / 30.0s maximum holding duration, synchronized with dynamic ATR-scaled barriers ($1.5 \times \text{ATR}_{10s}$ TP, $1.2 \times \text{ATR}_{10s}$ SL).
   - **Deterministic State Machine:** 6-state FSM (`IDLE`, `SUBMITTING_ENTRY`, `RESTING_ENTRY`, `POSITION_OPEN`, `RESTING_EXIT`, `RECORDING_TRADE`) preventing orphan orders or position desync.
   - **Telemetry:** Continuous serialization to `scratch/live_streaming_eval_ledger.jsonl`.
3. **Implementation Plan Ready:** The specification provides full drop-in code for `feature engineering scripts/inference/binance_testnet_engine.py` ready for the worker/implementer agent to deploy.

---

## 5. Verification Method

To independently verify this strategy:

1. **Contract & Unit Test Verification:**
   ```powershell
   pytest tests/test_leakage_and_features.py -v
   ```
   *Expected Result:* 48/48 tests pass, specifically verifying `TestMakerExecutionAndQueueDynamics` (Features 11–15).

2. **Purged Walk-Forward Maker Execution Simulation:**
   ```powershell
   python scripts/purged_walk_forward_backtest.py --fees-bps 4.0 --embargo-bars 3
   ```
   *Expected Result:* Walk-forward backtesting across $\ge 200$ clusters produces net-positive PnL under 4.0 bps VIP0 maker fees with Wilson CI lower bound $> 50.0\%$ and duration $\le 30.0\text{s}$.

3. **Verify Implementation Strategy Document:**
   Inspect [`testnet_execution_strategy.md`](file:///D:/backup/Binance%20data/.agents/teamwork/explorer_r3_testnet/testnet_execution_strategy.md) to verify complete class definitions (`BinanceTestnetClient`, `BinanceTestnetTradingEngine`), error code `-5022` handling, FSM transitions, and dynamic ATR barrier formulas.

4. **Invalidation Conditions:**
   - Any order execution path falling back to a `MARKET` taker entry upon code `-5022`.
   - Any holding duration exceeding 3 bars ($30.0\text{ seconds}$).
   - Any static barrier configuration not scaled to realized 10-second ATR.
