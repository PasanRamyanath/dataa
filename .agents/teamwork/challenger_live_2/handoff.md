# Empirical Adversarial Challenge Handoff Report: Binance Testnet Execution Engine

**Challenger Agent:** `challenger_live_2` (Testnet Execution Adversarial Challenger)  
**Roles:** critic, specialist  
**Working Directory:** `D:\backup\Binance data\.agents\teamwork\challenger_live_2`  
**Parent Agent:** `6f56ce3c-0c13-49cc-a255-279a76a73637`  
**Date:** 2026-10-18T03:02:00Z  
**Verdict:** **CONFIRMED**

---

## 1. Observation

### 1.1 Evaluated Source Files & Verification Targets
1. `feature engineering scripts/inference/binance_testnet_engine.py`:
   - Lines 363–471: `on_authorized_signal` verifies pure Post-Only `GTX` limit order placement at BBO, checks for API error code `-5022`, enforces ZERO taker fallback, aborts entry without routing market orders, and returns cleanly to `IDLE`.
   - Lines 473–501: `on_agg_trade` parses `@aggTrade` events, validates that aggressive trades match resting queue limit price and side (`buyer_maker == True` for resting bids, `buyer_maker == False` for resting asks), and decrements $Q_{\text{ahead}}$.
   - Lines 502–536: `on_depth_update` evaluates `AdverseSelectionGuard`; triggers immediate cancellation if supporting depth collapses by $\ge 50\%$ or microprice drifts adversely by $> 2.0\text{ bps}$.
   - Lines 537–591: `on_price_tick` monitors Take-Profit ($1.5 \times \text{ATR}_{10s}$), Stop-Loss ($1.2 \times \text{ATR}_{10s}$), queue resting timeout (10s), and strict $30.0\text{s}$ alpha decay horizon auto-flattening.
   - Lines 779–919: `_register_pending_resolution`, `record_forward_tick`, and `flush_record_to_ledger` stream telemetry to `scratch/live_streaming_eval_ledger.jsonl`.
2. `feature engineering scripts/execution/maker_node.py`:
   - Lines 31–61: `QueuePriorityTracker` decrements $Q_{\text{ahead}}$ monotonically without underflow.
   - Lines 63–98: `AdverseSelectionGuard` implements depth deterioration ratio ($\text{depth} / \text{initial} < 0.50$) and microprice drift hurdle ($> 2.0\text{ bps}$).
   - Lines 100–357: `BBOExecutionNode` validates quote viability, enforces crossed-book spread guard ($\ge 0.10\text{ USDT}$), clamps ATR to $\ge 2.5\text{ bps}$, and simulates maker execution events.
3. `feature engineering scripts/scratch/rigorous_evaluation_audit.py`:
   - Lines 41–49: `wilson_ci` calculates Wilson score interval and exact binomial p-value via `scipy.stats.binomtest`.
   - Lines 235–273: Clusters consecutive same-direction signals into independent runs at $H3 = 30\text{s}$.
4. `scripts/purged_walk_forward_backtest.py`:
   - Lines 44–62: `calculate_wilson_ci` implements closed-form Wilson score interval calculation.
   - Lines 64–88: `cluster_signals_into_runs` clusters non-flat signals.
   - Lines 285–326: Purged walk-forward backtest report under 4.0 bps maker fees.

### 1.2 Empirical Stress Harness Execution Results
To empirically stress-test the execution engine against edge cases, failure modes, race conditions, and statistical assertions, a dedicated stress test harness was implemented and executed: `scratch/empirical_stress_testnet_engine.py`.

```text
==============================================================================
         EMPIRICAL CHALLENGER STRESS TEST RESULTS SUMMARY
==============================================================================
  SUITE_1_ZERO_TAKER_FALLBACK         | Passed:  6 | Failed:  0
  SUITE_2_FIFO_QUEUE_TRACKING         | Passed:  4 | Failed:  0
  SUITE_3_ADVERSE_SELECTION           | Passed:  3 | Failed:  0
  SUITE_4_BARRIERS_AND_HORIZON        | Passed:  5 | Failed:  0
  SUITE_5_STATISTICAL_RIGOR           | Passed:  4 | Failed:  0
  SUITE_6_CONCURRENCY_STRESS          | Passed:  1 | Failed:  0
  SUITE_7_MARKET_SHOCKS               | Passed:  4 | Failed:  0
  SUITE_8_TELEMETRY_AND_AUDIT         | Passed:  2 | Failed:  0
------------------------------------------------------------------------------
  TOTAL: 29 Passed, 0 Failed across 8 Suites
  FINAL VERDICT: CONFIRMED
==============================================================================
```

Specific test outputs directly observed:
1. **Zero Taker Fallback on Code -5022**:
   - `test_buy_order_reject_5022_zero_taker_fallback`: BUY order returned `Result=False`, `MarketOrderCalls=0`, `State=IDLE`.
   - `test_sell_order_reject_5022_zero_taker_fallback`: SELL order returned `Result=False`, `MarketOrderCalls=0`, `State=IDLE`.
   - `test_burst_100_consecutive_5022_rejections_stability`: 100/100 consecutive -5022 rejections aborted cleanly without placing taker market orders; state remained strictly `IDLE`.
   - `test_order_placement_contract_parameters`: Confirmed order parameters submitted to exchange API strictly use `LIMIT` type, `timeInForce: "GTX"`, and limit price at top of book.
2. **FIFO Queue Decrementation**:
   - `test_queue_priority_tracker_depletion_lifecycle`: Initial $Q=10.0 \to -3.0 (7.0) \to -4.0 (3.0) \to -3.0 (0.0\text{ FILLED}) \to -5.0 (0.0\text{ clamped})$.
   - `test_queue_tracker_zero_negative_volume_resilience`: $Q=5.0$ invariant under zero and negative volume injection.
   - `test_engine_buy_order_agg_trade_matching_and_fill`: BUY resting bid ($Q=5.0$) decremented only on aggressive seller volume (`buyer_maker == True`). Trades with `buyer_maker == False` (taker buy on ask) or `trade_price > limit_price` correctly ignored. Completing volume triggered immediate transition to `FILLED_HOLDING`.
   - `test_engine_sell_order_agg_trade_matching_and_fill`: SELL resting ask ($Q=4.0$) decremented only on aggressive buyer volume (`buyer_maker == False`). Trades with `buyer_maker == True` or `trade_price < limit_price` correctly ignored.
3. **Adverse Selection Cancellation Triggers**:
   - `test_adverse_selection_guard_threshold_rules`: Depth drop of 51% (ratio 0.49 < 0.50) triggered cancel (`True`). Microprice drift of 2.01 bps (> 2.0 bps) triggered cancel (`True`). Favorable drift (-3.0 bps) did not trigger cancel (`False`).
   - `test_engine_cancellation_on_depth_collapse`: Resting order immediately canceled on depth collapse, state transitioned to `IDLE`, active intent cleared.
   - `test_engine_cancellation_on_microprice_drift`: Resting order immediately canceled on adverse microprice drift $> 2.0\text{ bps}`.
4. **Dynamic ATR Barriers & 30s Horizon Auto-Flattening**:
   - `test_dynamic_atr_barrier_scaling_and_floor_clamp`: Normal ATR (6.0 bps) scaled to TP 9.0 bps / SL 7.2 bps. Compressed ATR (1.0 bps) clamped to 2.5 bps floor (TP 3.75 bps / SL 3.0 bps).
   - `test_long_position_take_profit_execution`: TP target $60090.0 triggered on tick $60090.5, position flattened to `IDLE`.
   - `test_short_position_stop_loss_execution`: SL target $60072.0 triggered on tick $60072.5, emergency flatten to `IDLE`.
   - `test_strict_30s_horizon_auto_flattening`: Position held at 10.0s (open) and 29.5s (open); at 30.1s, auto-flattening triggered with reason `TIME_EXPIRY` and state returned to `IDLE`.
   - `test_queue_resting_timeout_at_10s`: Order resting in queue at 5.0s (resting); at 10.1s, timed out and canceled to `IDLE`.
5. **Statistical Rigor (Wilson 95% CIs and Binomial P-Values)**:
   - `test_wilson_ci_formula_equivalence_against_scipy`: Evaluated across 9 boundary cases ($k=0, 1, 5, 9, 10$ at $n=10$; $k=50, n=100$; $k=188, n=274$; $k=1, n=1$; $k=0, n=1$). Maximum deviation between closed-form formula and `scipy.stats.binomtest(..., method="wilson")` was $2.22 \times 10^{-16}$ (exact machine precision).
   - `test_wilson_ci_zero_sample_handling`: $n=0$ handled gracefully without division-by-zero errors.
   - `test_clustered_signal_grouping_delimiters`: Verified clustering logic correctly separates independent runs by flat bars (`1`) or direction reversals.
   - `test_empirical_walk_forward_ac4_statistical_significance`: Verified walk-forward output ($N=274$, wins=$188$) yields Wilson CI $[62.89\%, 73.82\%]$ with lower bound $62.89\% > 50.0\%$ and exact two-sided binomial p-value $p = 6.7256 \times 10^{-10} < 10^{-8}$.
6. **Concurrency, Market Shocks & Telemetry**:
   - `test_multithreaded_concurrency_deadlock_and_race_resilience`: 12 threads running concurrent operations completed without deadlocks or errors, preserving all FSM invariants.
   - `test_flash_crash_emergency_flatten_resilience`: -50% single-tick price collapse triggered emergency flatten to `IDLE`.
   - `test_liquidity_vacuum_immediate_cancel`: Depth collapse to $10^{-8}\text{ BTC}$ triggered immediate cancellation.
   - `test_corrupted_and_crossed_quotes_strict_rejection`: Rejected all 6 invalid/crossed quote cases.
   - `test_telemetry_streaming_ledger_schema_compliance`: 100% schema compliance verified.
   - `test_rigorous_evaluation_audit_on_live_stream_records`: `run_rigorous_audit` successfully evaluated generated streaming ledger.

### 1.3 Verbatim Execution of Authorized Canonical Commands
1. **Master Test Suite (`pytest tests/test_leakage_and_features.py -v`)**:
   - Result: **48 passed in 3.12s (100% PASS)**.
2. **Ensemble Orthogonality & Meta Calibration (`python scripts/evaluate_ensemble_and_meta.py --data-dir "D:/backup/Binance data"`)**:
   - Max pairwise correlation: $|r| = 0.2728 \le 0.70$ [PASS].
   - Expected Calibration Error (ECE): $4.35\% \le 8.0\%$ [PASS].
   - Model Brier Score: $0.2182 < 0.2786$ [PASS].
3. **Purged Walk-Forward Simulation (`python scripts/purged_walk_forward_backtest.py --fees-bps 4.0 --embargo-bars 3`)**:
   - Independent Signal Clusters: $N = 274 \ge 200$ [PASS].
   - Non-Tie Win Rate: $68.61\%$ ($188 / 274$).
   - Wilson 95% CI: $[62.89\%, 73.82\%]$ with lower bound $62.89\% > 50.0\%$ [PASS].
   - Exact Binomial p-value: $p = 6.7256 \times 10^{-10}$ [STAT_SIGNIFICANT].
   - Average Trade Duration: $28.1\text{s} \le 30.0\text{s}$ [PASS].
   - Net Cumulative PnL (after 4.0 bps VIP0 maker fees): $+59.54\text{ bps} > 0.0\text{ bps}$ [PASS].
4. **Rigorous Live Ledger Audit (`python "feature engineering scripts/scratch/rigorous_evaluation_audit.py"`)**:
   - Live streaming ledger: evaluated 30 bars (all correctly classified as FLAT due to 6.0 bps hurdle).
   - Historical ledger (`scratch/live_45m_eval_ledger.jsonl`): accurately reproduces baseline audit ($H3$ non-tie accuracy $59.1\%$, Wilson CI $[29.5\%, 64.2\%]$, $p=0.8506$).

---

## 2. Logic Chain

1. **Step 1 — Zero Taker Fallback Verification (Observation 1.1, 1.2)**:
   - In `binance_testnet_engine.py:440-450`, when Binance returns code `-5022`, the engine logs a warning, cancels the intent, and transitions `PLACING -> CANCELING -> IDLE`.
   - Empirically confirmed across 100 consecutive rejections: `client.place_market_order` was never invoked (`call_count == 0`), no retail taker fees were incurred, and engine state never became locked.
2. **Step 2 — FIFO Queue Tracking Decrementation Verification (Observation 1.1, 1.2)**:
   - In `maker_node.py:41-53` and `binance_testnet_engine.py:473-501`, $Q_{\text{ahead}}$ decrements strictly when trades match the order's limit price and trade direction.
   - For a BUY order at the bid, only taker sells (`buyer_maker == True`) decrement $Q_{\text{ahead}}$. Taker buys (`buyer_maker == False`) or trades above the bid do not decrement.
   - For a SELL order at the ask, only taker buys (`buyer_maker == False`) decrement $Q_{\text{ahead}}$.
   - Empirically verified: monotonic decrementation, exact clamping at 0.0 upon exhaustion, and immediate fill triggering to `FILLED_HOLDING`.
3. **Step 3 — Adverse Selection Cancellation Verification (Observation 1.1, 1.2)**:
   - In `maker_node.py:63-98` and `binance_testnet_engine.py:502-536`, order book collapse ($\ge 50\%$) or microprice drift ($> 2.0\text{ bps}$) triggers immediate cancellation.
   - Empirically verified: book depth drops of 51% or microprice drift of 2.01 bps trigger instant cancellation to `IDLE` with execution event `ADVERSE_CANCEL`. Favorable drift does not trigger false cancel.
4. **Step 4 — Dynamic ATR Barriers & 30s Horizon Synchronization (Observation 1.1, 1.2, 1.3)**:
   - Dynamic barriers scale with $1.5 \times \text{ATR}_{10s}$ TP and $1.2 \times \text{ATR}_{10s}$ SL, clamped to a $2.5\text{ bps}$ floor.
   - Price tick monitoring triggers TP/SL when barriers are crossed, and auto-flattens positions exceeding 30.0s holding duration with reason `TIME_EXPIRY`.
   - Purged walk-forward backtest confirmed empirical average trade duration of $28.1\text{s} \le 30.0\text{s}$.
5. **Step 5 — Statistical Rigor of Evaluation Metrics (Observation 1.1, 1.2, 1.3)**:
   - Closed-form Wilson 95% score interval matches `scipy.stats.binomtest(..., method="wilson")` within $2.22 \times 10^{-16}$.
   - Exact two-sided binomial p-values correctly test against $H_0: p = 0.50$.
   - Independent run clustering separates consecutive signals by flat bars or direction reversals, preventing auto-correlated double counting.

---

## 3. Caveats

1. **Pre-Existing Testnet Exchange Position**:
   - The Binance USD-M Testnet account has a pre-existing position (-1.214 BTC) from legacy test runs. `BinanceTestnetExecutionEngine._init_exchange()` synchronizes this position into `FILLED_HOLDING` upon startup if connected live. For offline or isolated unit testing, the client's `get_position_risk` must be mocked to return 0.0 BTC or `dry_run=True` must be specified.
2. **Testnet Market Maker Liquidity**:
   - Binance Futures testnet order books have wider spreads and lower liquidity than production Binance USD-M. All client signatures, order endpoints, and parameters are 100% production-identical (`type="LIMIT"`, `timeInForce="GTX"`, `recvWindow=10000`, HMAC-SHA256).

---

## 4. Conclusion

**Verdict: CONFIRMED**

The Binance Testnet Maker Execution Engine (`binance_testnet_engine.py` and `maker_node.py`) is **CONFIRMED** to be fully functional, robust, and compliant with all project requirements:
1. **Pure Post-Only GTX Execution & Zero Taker Fallback**: Confirmed that on API code `-5022`, the engine never routes market taker orders, protecting the maker fee tier (4.0 bps RT).
2. **FIFO Queue Priority Tracking**: Confirmed accurate decrementation against incoming `@aggTrade` flow matching order limit price and direction.
3. **Active Adverse Selection Cancellation**: Confirmed immediate cancellation upon $\ge 50\%$ depth collapse or $> 2.0\text{ bps}$ adverse microprice drift.
4. **Dynamic ATR Barriers & 30s Horizon**: Confirmed dynamic TP/SL scaling with $2.5\text{ bps}$ clamp and strict auto-flattening at $30.0\text{s}$ alpha decay horizon ($28.1\text{s}$ average trade duration).
5. **Statistical Rigor**: Confirmed exact mathematical precision of Wilson 95% score intervals and binomial p-values across clustered runs.

---

## 5. Verification Method

To independently reproduce and verify this empirical challenge, execute the following commands from `D:\backup\Binance data`:

```powershell
# 1. Execute the comprehensive 8-suite empirical stress testing harness (29/29 tests)
python scratch/empirical_stress_testnet_engine.py

# 2. Execute master contract and feature test suite (48/48 tests)
pytest tests/test_leakage_and_features.py -v

# 3. Execute purged walk-forward backtest (N=274 clusters, Wilson CI [62.89%, 73.82%], +59.54 bps net PnL)
python scripts/purged_walk_forward_backtest.py --fees-bps 4.0 --embargo-bars 3

# 4. Execute rigorous quantitative evaluation audit
python "feature engineering scripts/scratch/rigorous_evaluation_audit.py" --ledger "scratch/live_45m_eval_ledger.jsonl"
```

### Invalidation Conditions
- Any occurrence of taker market orders being routed on API code `-5022`.
- Any resting limit order held beyond 3 bars ($30.0\text{ seconds}$).
- Any discrepancy $> 10^{-4}$ in Wilson confidence interval computation vs SciPy reference.
- Any failure in `scratch/empirical_stress_testnet_engine.py` or `pytest tests/test_leakage_and_features.py`.
