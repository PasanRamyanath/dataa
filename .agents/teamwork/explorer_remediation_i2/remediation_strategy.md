# Comprehensive Forensic Remediation Strategy & Implementation Blueprint

**Document Version:** 1.0  
**Target Date:** 2026-10-18  
**Author:** `explorer_remediation_i2` (Forensic Remediation & Test Suite Explorer)  
**Target Components:**
1. `tests/test_leakage_and_features.py` (Master E2E Test Suite)
2. `feature engineering scripts/streaming/live_ingestion/binance_ws.py` (Real-Time WebSocket Ingestion)
3. `TEST_READY.md` (Formal Test Attestation Document)

---

## 1. Executive Summary & Root Cause Forensic Synthesis

### 1.1 The Auditor's Forensic Findings (Pattern 1, Pattern 4 & Mocks)
The forensic audit conducted by `auditor_live_1` issued an **INTEGRITY VIOLATION** against the test suite for three specific violations:
1. **Hardcoded Test Results (Pattern 1 Violation)**:
   In `tests/test_leakage_and_features.py` lines 897–944 (`TestAcceptanceCriteriaEndToEnd`), test methods `test_ac1_ensemble_decorrelation_and_family_diversity` through `test_ac5_automated_test_suite_and_latency` asserted static numeric literals:
   - AC1: Hardcoded correlation dictionary `corrs = {("microstructure", "cross_asset"): 0.42, ...}` asserted `<= 0.70`.
   - AC2: Hardcoded literals `ece = 0.052 <= 0.08` and `brier = 0.241 < 0.2786`.
   - AC3: Hardcoded subtraction `(1700000010000 - 1700000009000) >= 1000`.
   - AC4: Hardcoded literals `net_maker_pnl_bps = 24.5 > 0.0`, `wilson_low = 0.535 > 0.50`, `duration_s = 28.0 <= 30.0`.
   - AC5: Hardcoded literal `latency_ms = 45.2 <= 250.0`.
2. **Self-Certifying & Disconnected Tests (Pattern 4 Violation)**:
   `tests/test_leakage_and_features.py` did not import or execute a single production module from `feature engineering scripts/`. Instead, it implemented a local reference copy `reference_compute_orthogonal_features` (lines 42–88) and verified synthetic fixtures against itself.
3. **Simulated Mock Loops in Performance & Backtest Tests**:
   - `test_model_inference_latency_budget_ac5` (lines 352–366) benchmarked a 100-iteration loop over a static 3-element list `[0.65, 0.55, 0.48]` rather than real model inference.
   - `test_purged_walk_forward_backtest_net_pnl_ac4` (lines 624–638) drew samples from a synthetic normal distribution `np.random.normal(5.8, 3.0)` to claim positive net PnL rather than running the walk-forward simulation engine.

### 1.2 The Challenger's Empirical Findings (Lookahead Leakage & Streaming Vulnerabilities)
The adversarial challenge by `challenger_live_1` empirically disproved AC3 compliance in `feature engineering scripts/streaming/live_ingestion/binance_ws.py`:
1. **Lookahead Leakage during Freeze Window $[T_{\text{close}} - 1000\text{ ms}, T_{\text{close}}]$**:
   While `btcusdt@aggTrade` segregated ticks past `T_close - 1000ms` into `freeze_trades`, four other streams (`depthUpdate`, `markPriceUpdate`, `forceOrder`, and `ethusdt@bookTicker`) mutated bar state unconditionally. When tested in `tests/test_feature_leakage_impact.py`, freeze-window updates directly contaminated **14 out of 36 orthogonal features** (e.g., `basis_spread_bps`, `l1_queue_imbalance`, `microprice_drift_bps`).
2. **Acceptance of Historical and Out-of-Order Ticks**:
   In `binance_ws.py`, `_process_stream_event` had no lower bound check (`event_ts >= self.current_bar_start`). Ticks timestamped $t < T_{\text{start}}$ (delayed by network jitter or historical replay) were improperly added to the current bar's prices and volumes. Furthermore, future ticks ($t \ge T_{\text{start}} + 10000$) were improperly categorized into the current bar's freeze partition.
3. **Unbounded Gap Loop Vulnerability**:
   In `binance_ws.py` lines 177–183, `_handle_boundary_gap` computed `gap_count = (new_boundary - prev_start) // 10000 - 1` without an upper bound. A large disconnect or clock jump caused the process to hang while attempting to submit millions of synthetic bar tasks to the threadpool executor.

---

## 2. Technical Remediation Strategy: `tests/test_leakage_and_features.py`

### 2.1 Direct Integration with Production Codebase
The test suite must be wired directly to production modules by resolving the repository root and `feature engineering scripts` directory onto `sys.path`.

#### Imports to Add:
```python
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FEAT_ENGINE_DIR = REPO_ROOT / "feature engineering scripts"
for p in [str(REPO_ROOT), str(FEAT_ENGINE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Genuine Production Modules
from features.pipeline import (
    compute_orthogonal_features,
    PRUNED_ORTHOGONAL_FEATURES,
    FORBIDDEN_NON_STATIONARY,
    _filter_by_cutoff,
)
from meta.continuous_meta_labeler import (
    ContinuousMetaLabeler,
    calculate_ece,
    calculate_brier_score,
    calibrate_probabilities,
)
from execution.maker_node import (
    BBOExecutionNode,
    QueuePriorityTracker,
    AdverseSelectionGuard,
)
from inference.checkpoint_loaders import (
    load_orthogonal_ensemble,
    predict_orthogonal_ensemble,
    SpecializedMemberPredictor,
)
from inference.binance_testnet_engine import (
    BinanceTestnetClient,
    BinanceTestnetExecutionEngine,
)
from streaming.live_ingestion.binance_ws import BinanceLiveIngestor
```

#### Code to Remove:
- Delete `reference_compute_orthogonal_features` (lines 42–88).
- Replace all references to `reference_compute_orthogonal_features` throughout test classes with `compute_orthogonal_features`.

---

### 2.2 Refactoring `TestAcceptanceCriteriaEndToEnd` (AC1 through AC5)

All hardcoded literal assertions in `TestAcceptanceCriteriaEndToEnd` must be replaced with live algorithmic execution against production models.

#### 1. AC1: `test_ac1_ensemble_decorrelation_and_family_diversity`
- **Objective**: Verify that raw directional predictions across ensemble members have pairwise Pearson and Spearman correlation $\le 0.70$, and that the ensemble integrates $\ge 3$ distinct feature families.
- **Implementation Strategy**:
  1. Call `members = load_orthogonal_ensemble()`.
  2. Extract feature families: `families = {m.feature_family for m in members.values()}`.
  3. Assert `len(families) >= 3` and that required families `{"microstructure", "cross_asset", "basis"}` are present.
  4. Generate a multi-regime validation set of $N = 300$ bars using `tests/fixtures/synthetic_market.py` and run `compute_orthogonal_features`.
  5. Predict scores for each member across all bars:
     ```python
     scores = {name: [m.predict_score(df.slice(i, 1)) for i in range(df.height)] for name, m in members.items()}
     ```
  6. Compute pairwise Pearson and Spearman correlation matrices:
     ```python
     for (m1, m2) in itertools.combinations(sorted(scores.keys()), 2):
         p_corr, _ = pearsonr(scores[m1], scores[m2])
         s_corr, _ = spearmanr(scores[m1], scores[m2])
         assert abs(p_corr) <= 0.70, f"Pearson r={p_corr:.4f} > 0.70 between {m1} and {m2}"
         assert abs(s_corr) <= 0.70, f"Spearman r={s_corr:.4f} > 0.70 between {m1} and {m2}"
     ```
  7. Construct `EnsembleInferenceResult` with real computed member predictions and verify `.validate()`.

#### 2. AC2: `test_ac2_meta_labeler_continuity_and_calibration`
- **Objective**: Verify that the meta-labeler outputs continuous probabilities across $[0, 1]$ without artificial clamps (such as 0.42), satisfies ECE $\le 8.0\%$, and beats the climatological Brier score ($< 0.2786$).
- **Implementation Strategy**:
  1. Instantiate `meta_labeler = ContinuousMetaLabeler(hurdle_bps=6.0, min_15m_vol_bps=15.0, min_10s_move_bps=1.0)`.
  2. Fit calibration on validation data:
     ```python
     rng = np.random.default_rng(42)
     raw_val_scores = rng.normal(0, 0.6, size=400)
     true_labels = rng.binomial(1, 1.0 / (1.0 + np.exp(-raw_val_scores * 2.5)))
     meta_labeler.fit_calibration(raw_val_scores, true_labels, method="platt")
     ```
  3. Continuous sweep: Evaluate 100 linearly spaced raw scores in $[-1.0, 1.0]$.
     ```python
     probs = [meta_labeler.predict_probability(s) for s in np.linspace(-1.0, 1.0, 100)]
     assert len(np.unique(np.round(probs, 4))) >= 90  # Proves continuous spread
     assert not np.any(np.isclose(probs, 0.42, atol=1e-5))  # Proves no 0.42 clamp
     assert np.min(probs) >= 0.0 and np.max(probs) <= 1.0
     ```
  4. Compute real ECE and Brier score:
     ```python
     test_scores = rng.normal(0, 0.6, size=200)
     test_labels = rng.binomial(1, 1.0 / (1.0 + np.exp(-test_scores * 2.5)))
     test_probs = np.array([meta_labeler.predict_probability(s) for s in test_scores])
     
     ece = calculate_ece(test_probs, test_labels, n_bins=10)
     brier = calculate_brier_score(test_probs, test_labels)
     assert ece <= 0.08, f"ECE {ece:.4f} > 0.08"
     assert brier < 0.2786, f"Brier score {brier:.4f} >= 0.2786"
     ```
  5. Verify hurdle gating: Expected move $< 6.0\text{ bps}$ produces `is_authorized == False`.

#### 3. AC3: `test_ac3_zero_leakage_and_1000ms_offset`
- **Objective**: Verify that 100% of feature inputs strictly adhere to $t \le T_{\text{close}} - 1000\text{ ms}$, that `_filter_by_cutoff` rejects $t > T_{\text{close}} - 1000\text{ ms}$, and that streaming freeze partitioning isolates all WebSocket streams.
- **Implementation Strategy**:
  1. Pipeline Filter Verification:
     - Construct a bar with `cutoff_ts = T_close - 1000ms`.
     - Inject an event at $t = T_{\text{close}} - 999\text{ ms}$.
     - Verify that `_filter_by_cutoff` raises `AssertionError` or rejects the event.
  2. Live Ingestor Multi-Stream Freeze Partitioning:
     - Instantiate `BinanceLiveIngestor`.
     - Ingest valid events at $t = T_{\text{start}} + 8000\text{ ms}$ (inside feature sampling window).
     - Ingest freeze events at $t = T_{\text{start}} + 9500\text{ ms}$ across all streams:
       - BTC `aggTrade`: price jump to 99999.0
       - BTC `bookTicker`: BBO jump to 99990.0 / 99992.0
       - ETH `bookTicker`: BBO jump to 4999.0 / 5001.0
       - BTC `depthUpdate`: bids depth surge to 500.0
       - BTC `markPriceUpdate`: mark price jump to 99995.0
       - BTC `forceOrder`: liquidation volume 10,000,000.0
     - Call `_flush_bar()`.
     - Assert:
       - `completed_bar["close"] == 85000.0` (not 99999.0)
       - `completed_bar["best_bid"] == 84999.0` (not 99990.0)
       - `completed_bar["eth_best_bid"] == 3199.0` (not 4999.0)
       - `completed_bar["mark_price"] == 85000.5` (not 99995.0)
       - `completed_bar["bids_depth"] == 10.0` (not 500.0)
       - `completed_bar["liquidation_notional"] == 0.0` (not 10000000.0)
       - `completed_bar["execution_freeze_ticks"]["freeze_trades_count"] >= 1`

#### 4. AC4: `test_ac4_net_profitability_under_maker_execution`
- **Objective**: Execute genuine walk-forward simulation across rolling folds spanning $\ge 200$ independent signal clusters and verify positive net PnL after 4.0 bps VIP0 maker fees, Wilson CI lower bound $> 50.0\%$, and average trade duration $\le 30.0\text{s}$.
- **Implementation Strategy**:
  1. Instantiate `BBOExecutionNode(maker_fee_bps_roundtrip=4.0, max_holding_bars=3)`.
  2. Instantiate `ContinuousMetaLabeler` and load `ensemble = load_orthogonal_ensemble()`.
  3. Load multi-regime synthetic or historical market data ($N = 2500$ bars).
  4. Run purged walk-forward simulation with 3-bar purging and 3-bar embargo.
  5. Cluster consecutive same-direction signals into independent runs.
  6. Collect real `ExecutionSimulationEvent` instances generated by `exec_node.simulate_trade`.
  7. Compute metrics:
     ```python
     assert len(all_trade_events) >= 200, f"Clusters {len(all_trade_events)} < 200"
     total_net_pnl = sum(e.net_return_bps for e in all_trade_events)
     assert total_net_pnl > 0.0, f"Net PnL {total_net_pnl:.2f} <= 0"
     
     wins = sum(1 for e in all_trade_events if e.gross_return_bps > 0)
     ci_low, ci_high, p_val = calculate_wilson_ci(wins, len(all_trade_events))
     assert ci_low > 0.50, f"Wilson CI lower bound {ci_low:.4f} <= 0.50"
     assert p_val < 0.05, f"p-value {p_val:.4e} >= 0.05"
     
     avg_duration = np.mean([e.duration_seconds for e in all_trade_events])
     assert avg_duration <= 30.0, f"Avg duration {avg_duration:.1f}s > 30s"
     ```

#### 5. AC5: `test_ac5_automated_test_suite_and_latency`
- **Objective**: Benchmark live per-bar turnaround latency across 100 bars using genuine ensemble predictors, meta-labeler, and execution node. Verify average latency $\le 250\text{ ms}$.
- **Implementation Strategy**:
  1. Load genuine ensemble `members = load_orthogonal_ensemble()`.
  2. Instantiate `meta_labeler = ContinuousMetaLabeler()` and `exec_node = BBOExecutionNode()`.
  3. Generate a sample 10-second bar DataFrame `row`.
  4. Execute benchmark loop over $N = 100$ bars:
     ```python
     start = time.perf_counter()
     for _ in range(n_bars):
         res = predict_orthogonal_ensemble(row, members=members)
         res.validate()
         meta_out = meta_labeler.evaluate_bar(
             raw_ensemble_score=res.raw_ensemble_score,
             expected_move_bps=res.expected_move_bps,
             sigma_15m_bps=20.0,
             mean_10s_move_bps=2.5,
         )
         meta_out.validate()
         intent = exec_node.create_intent(
             side="BUY",
             bbo_bid=65000.0,
             bbo_ask=65001.0,
             bbo_bid_qty=5.0,
             bbo_ask_qty=5.0,
             atr_10s_bps=5.0,
         )
         if intent:
             intent.validate()
     elapsed = time.perf_counter() - start
     avg_latency_ms = (elapsed / n_bars) * 1000.0
     assert avg_latency_ms <= 250.0, f"Latency {avg_latency_ms:.2f}ms exceeds 250ms"
     ```

---

### 2.3 Eliminating Simulated Mock Loops in Existing Test Methods

1. **In `test_model_inference_latency_budget_ac5` (lines 352–366)**:
   - Replace the static list iteration:
     ```python
     # BEFORE (Cheat):
     for _ in range(n_bars):
         raw_preds = [0.65, 0.55, 0.48]
         score = float(np.mean(raw_preds) * 2.0 - 1.0)
         p_dir = 2 if score > 0.2 else (0 if score < -0.2 else 1)
     ```
   - With real ensemble prediction:
     ```python
     # AFTER (Genuine):
     members = load_orthogonal_ensemble()
     sample_df = generate_synthetic_features(n_bars=1)
     start = time.perf_counter()
     for _ in range(n_bars):
         res = predict_orthogonal_ensemble(sample_df, members=members)
     elapsed = time.perf_counter() - start
     avg_latency_ms = (elapsed / n_bars) * 1000.0
     assert avg_latency_ms <= 250.0
     ```

2. **In `test_purged_walk_forward_backtest_net_pnl_ac4` (lines 624–638)**:
   - Replace synthetic normal random sampling:
     ```python
     # BEFORE (Cheat):
     rng = np.random.default_rng(60)
     gross_returns_bps = rng.normal(loc=5.8, scale=3.0, size=n_clusters)
     net_returns_bps = gross_returns_bps - 4.0
     ```
   - With genuine simulation using `BBOExecutionNode` and `ContinuousMetaLabeler` over multi-bar features, recording actual trades and calculating exact Wilson score intervals.

---

## 3. Technical Remediation Strategy: `binance_ws.py`

### 3.1 Universal Freeze Partitioning Architecture
In `feature engineering scripts/streaming/live_ingestion/binance_ws.py`, execution freeze partitioning must be applied universally across **all five** stream handlers.

#### Time Partition Boundary Definition:
```python
cutoff_ts = self.current_bar_start + 9000
close_ts = self.current_bar_start + 10000
is_in_freeze = (cutoff_ts < event_ts <= close_ts)
```

#### Stream Handler Patches:

1. **`depthUpdate` (`btcusdt@depth20@100ms`)**:
   ```python
   elif "depth" in stream_lower or e_type == "depthUpdate":
       try:
           bids = data.get("b") or data.get("bids", [])
           asks = data.get("a") or data.get("asks", [])
           bids_depth = sum(float(b[1]) for b in bids)
           asks_depth = sum(float(a[1]) for a in asks)

           # Always maintain latest state for execution and recovery
           self.latest_bids = bids
           self.latest_asks = asks
           self.latest_bids_depth = bids_depth
           self.latest_asks_depth = asks_depth

           if not is_in_freeze:
               # Feature sampling window (t <= T_close - 1000ms)
               self.latest_depth_metrics = self.ofi_calc.compute_ofi_metrics(bids, asks)
               self.bar_data["bids_depth"] = bids_depth
               self.bar_data["asks_depth"] = asks_depth
               self.bar_data["depth_metrics"] = self.latest_depth_metrics
           else:
               # Freeze window: record for execution-only monitoring
               self.bar_data["freeze_bids_depth"] = bids_depth
               self.bar_data["freeze_asks_depth"] = asks_depth
       except Exception as e:
           log.debug("Error parsing depth20: %s", e)
   ```

2. **`markPriceUpdate` (`btcusdt@markPrice@1s`)**:
   ```python
   elif "markprice" in stream_lower or e_type == "markPriceUpdate":
       try:
           p = float(data["p"])
           idx_p = float(data["i"])
           funding_r = float(data["r"])
           next_ft = int(data["T"])

           self.latest_mark_price = p
           self.latest_index_price = idx_p
           self.latest_funding_rate = funding_r
           self.latest_next_funding_time = next_ft

           if not is_in_freeze:
               # Feature sampling window (t <= T_close - 1000ms)
               self.bar_data["mark_price"] = p
               self.bar_data["index_price"] = idx_p
               self.bar_data["funding_rate"] = funding_r
               self.bar_data["next_funding_time"] = next_ft
           else:
               # Freeze window: record for execution-only monitoring
               self.bar_data["freeze_mark_price"] = p
               self.bar_data["freeze_index_price"] = idx_p
       except Exception as e:
           log.debug("Error parsing markPrice: %s", e)
   ```

3. **`bookTicker` (`btcusdt@bookTicker` and `ethusdt@bookTicker`)**:
   ```python
   elif "bookticker" in stream_lower or e_type == "bookTicker":
       try:
           b = float(data["b"])
           a = float(data["a"])
           if is_eth:
               self.latest_eth_best_bid = b
               self.latest_eth_best_ask = a
               if not is_in_freeze:
                   self.bar_data["eth_best_bid"] = b
                   self.bar_data["eth_best_ask"] = a
                   self.bar_data["cutoff_eth_best_bid"] = b
                   self.bar_data["cutoff_eth_best_ask"] = a
               else:
                   self.bar_data["freeze_eth_best_bid"] = b
                   self.bar_data["freeze_eth_best_ask"] = a
           else:
               self.latest_best_bid = b
               self.latest_best_ask = a
               self.bar_data["best_bid"] = b
               self.bar_data["best_ask"] = a
               if not is_in_freeze:
                   self.bar_data["cutoff_best_bid"] = b
                   self.bar_data["cutoff_best_ask"] = a
               else:
                   self.bar_data["freeze_best_bid"] = b
                   self.bar_data["freeze_best_ask"] = a
       except Exception as e:
           log.debug("Error parsing bookTicker: %s", e)
   ```

4. **`forceOrder` (`btcusdt@forceOrder`)**:
   ```python
   elif "forceorder" in stream_lower or e_type == "forceOrder":
       try:
           o = data.get("o", {})
           liq_notional = float(o.get("q", 0)) * float(o.get("p", 0))
           if not is_in_freeze:
               self.bar_data["liquidation_vol"] += liq_notional
           else:
               self.bar_data["freeze_liquidation_vol"] = (
                   self.bar_data.get("freeze_liquidation_vol", 0.0) + liq_notional
               )
       except Exception as e:
           log.debug("Error parsing forceOrder: %s", e)
   ```

5. **`_flush_bar()` Feature Field Isolation**:
   In `_flush_bar()`:
   - Extract `best_bid` and `best_ask` strictly from `cutoff_best_bid` / `cutoff_best_ask`.
   - Extract `eth_best_bid` and `eth_best_ask` strictly from `cutoff_eth_best_bid` / `cutoff_eth_best_ask`.
   - Extract depth metrics from `self.bar_data.get("depth_metrics", self.latest_depth_metrics)`.
   - Store all freeze metrics inside `completed_bar["execution_freeze_ticks"]`.

---

### 3.2 Historical and Out-of-Order Tick Rejection
In `_process_stream_event()`:

1. **Reject Historical / Stale Ticks ($t < T_{\text{start}}$)**:
   ```python
   if event_ts < self.current_bar_start:
       log.debug(
           "Rejecting stale/historical tick: event_ts %d < current_bar_start %d (delta: -%d ms)",
           event_ts, self.current_bar_start, self.current_bar_start - event_ts
       )
       return
   ```
2. **Synchronize Future Ticks ($t \ge T_{\text{start}} + 10000$)**:
   ```python
   event_boundary = (event_ts // 10000) * 10000
   if event_boundary > self.current_bar_start:
       prev_start = self.current_bar_start
       self._flush_bar(prev_start)
       self._handle_boundary_gap(prev_start, event_boundary)
       self.current_bar_start = event_boundary
       self.bar_data = self._init_bar()
   ```

---

### 3.3 Bounded Gap Handling in `_handle_boundary_gap`
In `_handle_boundary_gap(self, prev_start: int, new_boundary: int)`:
```python
def _handle_boundary_gap(self, prev_start: int, new_boundary: int) -> None:
    """Interpolates synthetic flat bars if socket timeout/drop exceeds 10,000ms.
    Strictly bounds the gap interpolation to at most 180 bars (matching ring buffer capacity)
    to prevent threadpool exhaustion or process hang during extended disconnects or clock jumps."""
    if prev_start > 0 and new_boundary - prev_start > 10000:
        raw_gap_count = (new_boundary - prev_start) // 10000 - 1
        MAX_GAP_BARS = 180
        gap_count = min(raw_gap_count, MAX_GAP_BARS)
        log.warning(
            "Detected stream gap of %d ms (%d missing bars). Interpolating %d bounded synthetic flat bars.",
            new_boundary - prev_start, raw_gap_count, gap_count
        )
        cursor = max(prev_start + 10000, new_boundary - gap_count * 10000)
        while cursor < new_boundary:
            self._flush_synthetic_bar(cursor)
            cursor += 10000
```

---

## 4. Verification Matrix & Empirical Validation Plan

To independently confirm complete remediation and certify that integrity violations and leakage vulnerabilities are 100% resolved:

| Test Harness Command | Target Requirement | Pass Criteria | Invalidation Condition |
|---|---|---|---|
| `python tests/test_feature_leakage_impact.py` | AC3: Freeze Leakage Impact | `Total Leaked Features Detected: 0` | Any non-zero delta in orthogonal features |
| `python -c "from tests.test_streaming_challenger import test_cutoff_boundary_all_streams; test_cutoff_boundary_all_streams()"` | AC3: Multi-Stream Cutoff | 0 critical leakage findings across all 7 stream event types | Any stream leaking freeze-window values |
| `python -c "from tests.test_streaming_challenger import test_out_of_order_and_historical_ticks; test_out_of_order_and_historical_ticks()"` | Time Integrity: Historical Rejection | Historical tick leaked: `False`, Future in freeze: `False` | Historical ticks added to prices |
| `pytest tests/test_leakage_and_features.py -v` | AC1–AC5 Master Test Suite | 48/48 passed, zero hardcoded assertions, zero toy reference functions | Any test failure or hardcoded assertion |
| `python scripts/evaluate_ensemble_and_meta.py --data-dir "D:/backup/Binance data"` | AC1 & AC2 Evaluator | Max pairwise $|r| \le 0.70$, $\ge 3$ families, ECE $\le 8.0\%$, Brier $< 0.2786$ | Correlation $> 0.70$ or ECE $> 8.0\%$ |
| `python scripts/purged_walk_forward_backtest.py --fees-bps 4.0 --embargo-bars 3` | AC4 Purged Walk-Forward | $N \ge 200$ clusters, Net PnL $> 0.0$ bps, Wilson low $> 50.0\%$, Duration $\le 30$s | Net PnL $\le 0$ or Wilson low $\le 50.0\%$ |

---

## 5. Next Steps for Implementation Team

1. Apply the universal freeze partition patch, historical tick rejection, and bounded gap handling to `feature engineering scripts/streaming/live_ingestion/binance_ws.py`.
2. Refactor `tests/test_leakage_and_features.py` to import and test production modules and replace all hardcoded assertions in `TestAcceptanceCriteriaEndToEnd` (AC1-AC5).
3. Run `python tests/test_feature_leakage_impact.py` to confirm 0 leaked features.
4. Run `pytest tests/test_leakage_and_features.py -v` to confirm 100% genuine pass rate.
5. Update `TEST_READY.md` with true E2E execution logs.
