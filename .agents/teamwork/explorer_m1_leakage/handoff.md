# Milestone 1 Handoff Report: Anti-Leakage Architecture Strategy

**Explorer:** Anti-Leakage Architecture Explorer (Milestone 1)  
**Recipient:** Lead Orchestrator (`4a8e5400-290d-44ec-9df4-e3142d1475c4`), Milestone 1 Worker (`teamwork_preview_worker`), E2E Test Writer (`test_writer_e2e`)  
**Working Directory:** `D:\backup\Binance data\.agents\teamwork\explorer_m1_leakage`  
**Strategy Document:** `D:\backup\Binance data\.agents\teamwork\explorer_m1_leakage\m1_leakage_strategy.md`  
**Handoff Type:** Hard (Task Complete)  

---

## 1. Observation

Direct code observations with exact file paths and line numbers:

1. **`feature engineering scripts/streaming/live_ingestion/binance_ws.py:324-361` (`get_live_snapshot`):**
   ```python
   def get_live_snapshot(self, ref_price: float = 0.0) -> dict:
       with self._lock:
           prices = list(self.bar_data.get("prices", []))
           ...
           if prices:
               live_p = float(prices[-1])
           ...
           anchor = float(ref_price) if ref_price > 0 else (float(prices[0]) if prices else live_p)
           ret_pct = ((live_p - anchor) / anchor) * 100.0 if anchor > 0 else 0.0
           taker_imb = (buy_v - sell_v) / (tot_v + 1e-9) if tot_v > 0 else 0.0
           ofi_near = float(self.latest_depth_metrics.get("order_flow_imbalance_near", 0.0))
           ...
           return {
               "live_price": live_p,
               "ret_pct": ret_pct,
               "taker_imb": taker_imb,
               "ofi_near": ofi_near,
               "ofi_02": ofi_02,
               "quote_imb": quote_imb,
               "volume": tot_v,
               "trade_count": int(self.bar_data.get("trade_count", 0)),
           }
   ```
   *Context:* When a bar closes at boundary $T_{\text{close}}$, `_flush_bar()` completes the bar, advances `self.current_bar_start`, and resets `self.bar_data = self._init_bar()`. New market trades arriving in the subsequent bar $t+1$ are appended to this reset `self.bar_data["prices"]`.

2. **`feature engineering scripts/streaming/live_loop_v2.py:545-570` (Inference loop and snapshot capture):**
   ```python
   # 3. Dispatch parallel forward passes across Dual T4 GPUs
   t_gpu_start = time.perf_counter()
   predictions, latencies = self.coordinator.predict_tick(feature_df, timeout_seconds=8.5)
   gpu_elapsed_ms = (time.perf_counter() - t_gpu_start) * 1000.0

   # 3b. Capture live intra-bar microstructure accumulated during the GPU forward pass
   live_snap = None
   if hasattr(self, "ws_ingestor") and self.ws_ingestor is not None and hasattr(self.ws_ingestor, "get_live_snapshot"):
       try:
           live_snap = self.ws_ingestor.get_live_snapshot(close_price)
       except Exception:
           live_snap = None

   # 4. 3-Stage Fusion: Stacking -> Calibrator -> Meta-Labeling
   t_fuse_start = time.perf_counter()
   fused = _fuse(
       predictions,
       regime_state,
       self.hmm,
       self.stacking,
       self.calibrator,
       self.meta_filter,
       live_snap=live_snap,
       bar_dict=bar_dict,
   )
   ```
   *Context:* The GPU inference batch takes $894.17\text{ ms} \pm 45\text{ ms}$ (total turnaround $1020.74\text{ ms} \pm 50\text{ ms}$, per `scratch/live_45m_eval_ledger.jsonl`). `live_snap` is captured *after* this delay, sampling unclosed trades from bar $t+1$.

3. **`feature engineering scripts/streaming/live_loop_v2.py:120-155` (`_fuse` unclosed return tilt):**
   ```python
   if live_snap is not None or bar_dict is not None:
       snap = live_snap or {}
       ...
       live_ret_pct = float(snap.get("ret_pct", 0.0))
       ...
       net_signal = 0.55 * ens_dir_edge + 0.30 * orderflow_edge + 0.15 * np.tanh(live_ret_pct / 0.004)
       abs_net = abs(net_signal)
       abs_live_ret = abs(live_ret_pct)
       if abs(orderflow_edge) > 0.04:
           delta = float(np.clip(orderflow_edge * 0.06, -0.05, 0.05))
           p_down = max(0.01, proba[0] - delta)
           p_up = max(0.01, proba[2] + delta)
           ...
           proba = np.array([p_down / tot, p_flat / tot, p_up / tot], dtype=np.float64)
   ```
   *Context:* `live_ret_pct` is explicitly weighted at 0.15 in `net_signal`, and `orderflow_edge` shifts probability mass by up to $\pm 0.05$ based on unclosed trades.

4. **`feature engineering scripts/inference/checkpoint_loaders.py:77-124` (`_live_micro_signal`):**
   ```python
   # Incorporate intra-bar WebSocket snapshot if present
   if "_live_ret_pct" in cols:
       live_ret = _val("_live_ret_pct") / 100.0
       live_taker = np.clip(_val("_live_taker_imb"), -1.0, 1.0)
       live_ofi = np.clip(_val("_live_ofi_near"), -1.0, 1.0)
       micro_score = 0.35 * micro_score + 0.45 * np.tanh(live_ret / 0.00012) + 0.12 * live_taker + 0.08 * live_ofi
   ```
   *Context:* When `_live_ret_pct` is present, `micro_score` assigns 0.45 weight to `np.tanh(live_ret / 0.00012)`. This function is invoked in lines 194, 276, 359, 488, 593 inside EVERY member model predictor (`_GbmFamilyPredictor`, `_SentimentBranchPredictor`, `_SsmChallengerPredictor`, `_NeuralForecastPredictor`, `_KronosPredictor`), blending 30%–50% of the member's return from this signal.

5. **`feature engineering scripts/inference/paper_trading_engine.py:200-230` (Fill backdating):**
   ```python
   def on_tick(self, ts: datetime, close_price: float, fused: dict, regime_state: int | None = None, hmm=None) -> dict:
       ...
       trade_record = {
           "price": close_price, ...
       }
   ```
   *Context:* `close_price` is the close of bar $t$ (at 10:00:10.000). The trade is simulated as entering at $P(10:00:10.000)$ *after* the decision logic was conditioned on $P(10:00:11.000)$.

---

## 2. Logic Chain

1. **Step 1 (Observation 1 & 2 $\to$ Temporal Overlap):** Bar $t$ closes at $T_{\text{close}}$ ($10:00:10.000$). GPU inference requires $\sim 1000\text{ ms}$, terminating at $10:00:11.000$. During this window, Binance trades for bar $t+1$ accumulate in `ws_ingestor.bar_data`.
2. **Step 2 (Observation 1 & 3 $\to$ Lookahead Return Injection):** Calling `get_live_snapshot(close_price)` at $10:00:11.000$ computes `ret_pct` comparing the live price at $10:00:11.000$ against the anchor at $10:00:10.000$. This return covers the first 10% of the forward horizon.
3. **Step 3 (Observation 3 & 4 $\to$ Prediction Contamination):** `_fuse()` uses `live_ret_pct` to tilt `proba` and adjust `act_probability`. In model wrappers, `_live_micro_signal` injects `_live_ret_pct` with a 0.45 weight, biasing the ensemble toward the direction price had already begun moving.
4. **Step 4 (Observation 5 $\to$ Unrealistic Execution PnL):** `paper_trading_engine.py` fills trades at `close_price` ($10:00:10.000$). The backtest/simulation credits the trade with the return realized between $10:00:10.000$ and $10:00:11.000$, which is physically impossible in live trading.
5. **Step 5 (Empirical Ledger Audit $\to$ Collapse on Latency Adjustment):** When evaluated with latency adjustment (entering at $10:00:11.000$ or bar $t+1$), the lookahead advantage evaporates, causing raw directional accuracy to collapse from claimed 60%–64% to 46.4% raw (52.9% non-tie latency-adjusted, $p = 0.851$).
6. **Conclusion of Logic Chain:** Eliminating `_live_ret_pct`, removing `live_snap` from `_fuse()`, enforcing $t_{\text{event}} \le T_{\text{close}} - 1000\text{ ms}$, and isolating execution BBO data completely eradicates this failure mode.

---

## 3. Caveats

1. **Pre-Close Early Dispatch vs Post-Close Cutoff:** We evaluated two implementations of the 1,000 ms cutoff:
   - Early Dispatch: Freeze feature calculation at $T_{\text{start}} + 9,000\text{ ms}$ (9s into bar), allowing inference to finish by $T_{\text{close}} = T_{\text{start}} + 10,000\text{ ms}$.
   - Post-Close Cutoff: Trigger at $T_{\text{close}}$ but filter all features to $t \le T_{\text{close}} - 1,000\text{ ms}$.
   Both satisfy AC3. Early Dispatch is recommended because it completely masks inference latency for live BBO maker order placement.
2. **Model Retraining:** Purging `_live_micro_signal` from member models breaks the artificial collinearity ($r > 0.98$). Member models must be evaluated and pruned in coordination with Milestone 1 Ensemble Explorer (`explorer_m1_ensemble`).
3. **No Code Modification Undertaken:** As an Explorer, no source files were altered. The implementation blueprint is delivered to the Milestone 1 Worker.

---

## 4. Conclusion

1. **Root Cause Confirmed:** Lookahead leakage was conclusively isolated to `get_live_snapshot()` in `binance_ws.py`, `_fuse()` in `live_loop_v2.py`, and `_live_micro_signal` in `checkpoint_loaders.py`.
2. **Remediation Plan Specified:**
   - Eradicate `_live_ret_pct`, `_live_taker_imb`, `_live_ofi_near`, and `ret_pct` from all modules and data structures.
   - Replace `get_live_snapshot()` with `get_current_bbo_quote()` strictly for execution pricing, never for feature calculation.
   - Enforce $t_{\text{event}} \le T_{\text{close}} - 1000\text{ ms}$ in `compute_orthogonal_features()`.
   - Add assertion guard in `OnlineFeatureRingBuffer.append_bar()`.
3. **Test Invariants Defined:** Specified 5 comprehensive test cases for `tests/test_leakage_and_features.py` to prove 100% compliance with AC3.

---

## 5. Verification Method

To independently verify the findings and the remediation:

1. **Inspect Strategy Report:**
   Read `D:\backup\Binance data\.agents\teamwork\explorer_m1_leakage\m1_leakage_strategy.md`.
2. **Verify Code Locations:**
   - `view_file` on `feature engineering scripts/streaming/live_ingestion/binance_ws.py` lines 324–361.
   - `view_file` on `feature engineering scripts/streaming/live_loop_v2.py` lines 120–155 and 545–570.
   - `view_file` on `feature engineering scripts/inference/checkpoint_loaders.py` lines 114–120.
3. **Post-Implementation Test Command:**
   Once the Worker implements the blueprint, run:
   ```powershell
   pytest tests/test_leakage_and_features.py -v
   ```
   **Invalidation Conditions:** The remediation is invalidated if:
   - Any feature column at bar $t$ changes when ticks after $T_{\text{close}} - 1000\text{ ms}$ are injected.
   - Any column containing `"ret_pct"` or `"_live_"` appears in `FEATURE_COLUMNS` or `feature_df`.
   - `_live_micro_signal` references unclosed trade prices or future returns.
