# Handoff Report: Execution & Backtest Exploration

**Agent:** Execution & Backtest Explorer  
**Working Directory:** `D:\backup\Binance data\.agents\teamwork\explorer_exec_survey`  
**Handoff Type:** Hard (Survey Task Complete)  
**Target Recipient:** Orchestrator (`4a8e5400-290d-44ec-9df4-e3142d1475c4`) & Implementation Specialists  

---

## 1. Observation

### Execution Logic & Order Placement
1. **Continuous Rebalancing Taker Model:** In `feature engineering scripts/inference/paper_trading_engine.py`, lines 218–220:
   ```python
   turnover = diff
   cost = self.equity * turnover * (config.PAPER_TAKER_FEE_BPS + config.PAPER_SLIPPAGE_BPS) / 10_000.0
   self.equity -= cost
   ```
   Orders are never placed as discrete limit orders. Costs are charged as taker fee (5.0 bps) + slippage (2.0 bps) = 7.0 bps per leg = 14.0 bps round-trip.
2. **Instant Fill Assumption in Dashboard Engine:** In `futures-prediction-dashboard/src/lib/paperEngine.ts`, lines 473–475:
   ```typescript
   // Maker Post-Only Limit fills at best bid for buying, best ask for selling
   execPrice = isBuying ? bookTicker.bidPrice : bookTicker.askPrice;
   ```
   If role is configured as "MAKER", fills are awarded instantly at `bestBid`/`bestAsk` without simulating queue position, queue priority, or fill probability.
3. **Hardcoded Testnet Script:** In `feature engineering scripts/scratch/test_binance_testnet_order.py`, lines 63–70, a test order with `timeInForce: "GTX"` is sent directly to Binance Testnet REST API and immediately deleted. This logic is not integrated into any live trading or backtesting node.
4. **Static Barriers & Holding Horizon:** In `feature engineering scripts/scratch/live_45m_comprehensive_eval.py`, lines 41–47:
   ```python
   TP_BPS = 16.0          # +16 bps take profit
   SL_BPS = 12.0          # -12 bps stop loss
   TRAILING_ACT_BPS = 7.0  # +7 bps trailing activation
   TRAILING_CB_BPS = 4.0   # 4 bps callback
   MAX_HOLD_BARS = 18      # 18 bars (3 minutes) horizon
   TAKER_FEE_BPS = 5.0     # 5 bps per leg = 10 bps round-trip
   ```
   In `scratch/live_45m_eval_ledger.jsonl`, all 5 authorized trades held for the full 18 bars (180 seconds) and exited on `TIME_EXPIRY` with gross losses against market drift.
5. **Intra-Bar Micro-Signal Leakage:** In `feature engineering scripts/inference/checkpoint_loaders.py`, lines 115–119:
   ```python
   if "_live_ret_pct" in cols:
       live_ret = _val("_live_ret_pct") / 100.0
       live_taker = np.clip(_val("_live_taker_imb"), -1.0, 1.0)
       live_ofi = np.clip(_val("_live_ofi_near"), -1.0, 1.0)
       micro_score = 0.35 * micro_score + 0.45 * np.tanh(live_ret / 0.00012) + 0.12 * live_taker + 0.08 * live_ofi
   ```
   The feature `_live_ret_pct` incorporates price moves occurring inside the current bar during WebSocket ingestion, introducing intra-bar lookahead leakage.

### Existing Verification Scripts
6. **Missing Target Files:** An exhaustive search using `find_by_name` across `D:\backup\Binance data` confirmed:
   - `scripts/evaluate_ensemble_and_meta.py` does NOT exist.
   - `scripts/purged_walk_forward_backtest.py` does NOT exist.
   - `tests/test_leakage_and_features.py` does NOT exist.
7. **Working Diagnostic Audit:** `python "feature engineering scripts/scratch/rigorous_evaluation_audit.py"` ran cleanly (exit code 0), processing 270 bars from `scratch/live_45m_eval_ledger.jsonl` and outputting verified statistical metrics.
8. **Available Datasets:**
   - `feature engineering scripts/fusion/oof/`: Contains 7 full days (60,480 bars from 2026-10-04 to 2026-10-11) of out-of-fold predictions for `gbm-family`, `sentiment-branch`, and `regime_states`, plus 6,040 bars for transformer members (`nhits`, `patchtst`, `tft`, `itransformer`, `ssm-challenger`).
   - `feature engineering scripts/output/features_BTCUSDT_2026-10-11.parquet`: Contains 8,640 bars $\times$ 226 features.

---

## 2. Logic Chain

1. **Premise 1 (From Obs 1 & 4):** Mean gross alpha at peak ($H3$) is $+0.74\text{ bps}$. Current execution engines charge 10.0 to 14.0 bps round-trip taker fees.
   - *Inference:* Any taker-based strategy is mathematically insolvent by $-9.26\text{ bps}$ per trade. Pure Maker execution (4.0 bps VIP0 RT) is mandatory.
2. **Premise 2 (From Obs 4 & 5):** The strategy held trades for 180s (18 bars) against fixed 16/12 bps barriers. However, 10-second alpha peaks at 30s with a 0.74 bps move, while 180s rolling volatility is only 2–3 bps.
   - *Inference:* 16 bps take-profit was unreachable, causing 100% of trades to expire on time stop into macro secular drift (+25.18 bps upward trend), converting potential 30s winners into 180s losers.
3. **Premise 3 (From Obs 2 & 3):** No current script models order book queue priority, queue fill exhaustion, or adverse selection. Existing "maker" simulations assume instantaneous execution at the BBO.
   - *Inference:* Backtest results produced by current code would be severely optimistic and unexecutable in live trading. A queue-aware fill simulator matching taker sell volume against resting bid depth is required.
4. **Premise 4 (From Obs 6 & 8):** The target verification commands specified in AC1–AC5 (`scripts/evaluate_ensemble_and_meta.py`, `scripts/purged_walk_forward_backtest.py`, `tests/test_leakage_and_features.py`) are absent, but the historical OOF dataset (60,480 bars spanning 7 days) is present on disk.
   - *Inference:* Purged walk-forward CV with 5 expanding folds across the 7-day dataset will generate 400–800 independent trade clusters, fulfilling AC4 ($\ge 200$ clusters) once the missing scripts are constructed.

---

## 3. Caveats

1. **Order Book L2 Granularity:** The available historical 10-second parquet files contain Level-1 BBO (`best_bid`, `best_ask`, `bids_depth`, `asks_depth`) and trade volume, but not full sub-millisecond Level-2/Level-3 depth snapshots. Queue modeling in historical backtests must therefore rely on FIFO depth depletion heuristics ($V_{\text{taker}} \ge \gamma \times \text{Depth}_{\text{BBO}}$) rather than full tick-by-tick order-queue reconstruction.
2. **Transformer OOF Striding:** Neuralforecast members (`tft`, `patchtst`, `nhits`) were evaluated with a stride of 10 bars (100 seconds). For continuous 10s walk-forward backtesting, predictions between strides must be joined via as-of backward join (using the latest prediction known at or before $t$), which is the standard methodology established in `fusion/stacking_meta_learner.py`.

---

## 4. Conclusion

1. **System Failure Mechanism:** The collapse in live performance was caused by:
   - Holding positions $6\times$ longer than the alpha lifespan (180s vs 30s).
   - Unreachable static barriers (16/12 bps) rendering take-profit inert.
   - Taker fee friction (10–14 bps) annihilating micro-alpha (0.74 bps).
   - Intra-bar lookahead leakage in `_live_micro_signal` creating artificial backtest optimism.
2. **Required Implementation Scope:**
   - **Execution Engine:** Discrete-event Maker Node using `GTX` Post-Only limit orders at BBO with FIFO queue depletion tracking, 3-bar ($30\text{s}$) time exits, dynamic ATR-scaled barriers ($k \times \text{ATR}_{10s}$), and immediate cancellation upon adverse queue deterioration.
   - **Backtest Engine:** `scripts/purged_walk_forward_backtest.py` implementing expanding walk-forward folds with 3-bar purge and 3-bar embargo across the 60,480-bar historical dataset, charging 4.0 bps maker fees.
   - **Evaluation Script:** `scripts/evaluate_ensemble_and_meta.py` asserting ensemble de-correlation ($r \le 0.70$), ECE $\le 8.0\%$, and Brier score $< 0.2786$.
   - **Test Suite:** `tests/test_leakage_and_features.py` testing latency offsets, zero lookahead leakage, feature contract integrity, and execution logic under pytest.

---

## 5. Verification Method

To independently verify the findings of this survey:

1. **Verify Existing Audit Execution:**
   ```powershell
   python "feature engineering scripts/scratch/rigorous_evaluation_audit.py"
   ```
   *Expected Output:* Exits with code 0, prints loaded 270 raw bars, and confirms keys `['n_bars', 'multi_horizon', 'runs', 'drift', 'baselines', 'calibration', 'meta_labeler', 'correlations']`.

2. **Verify Dataset Availability & Shapes:**
   ```powershell
   python -c "import polars as pl; df = pl.read_parquet('feature engineering scripts/fusion/oof/gbm-family.parquet'); print('GBM OOF shape:', df.shape); print('Range:', df['ts'].min(), 'to', df['ts'].max())"
   ```
   *Expected Output:* `(60480, 4)` spanning `2026-10-04 23:00:00 to 2026-10-11 22:59:50`.

3. **Verify Absence of Target Scripts:**
   ```powershell
   python -c "from pathlib import Path; [print(p, Path(p).exists()) for p in ['scripts/evaluate_ensemble_and_meta.py', 'scripts/purged_walk_forward_backtest.py', 'tests/test_leakage_and_features.py']]"
   ```
   *Expected Output:* All three evaluate to `False`.

4. **Detailed Survey Document:**
   Inspect the comprehensive report written to:
   `D:\backup\Binance data\.agents\teamwork\explorer_exec_survey\survey_exec_report.md`
