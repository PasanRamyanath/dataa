# Exhaustive Survey Report: Execution Logic, Backtest Engine, Testing Suites & Evaluation Scripts

**Author:** Execution & Backtest Explorer  
**Date:** 2026-10-18  
**Project:** BTCUSDT 10-Second Alpha & Execution Overhaul  
**Target Repository:** `D:\backup\Binance data`  
**Status:** Comprehensive Baseline & Architectural Assessment  

---

## 1. Executive Summary & Root Cause Diagnosis

An exhaustive inspection of the codebase in `D:\backup\Binance data` reveals that the empirical collapse of the live trading system (from claimed 60%–64% accuracy down to 46.4% raw, 52.9% latency-adjusted, and net negative PnL across all authorized signals) stems from severe structural defects in the execution logic, the lack of realistic microstructure backtesting, flawed barrier/holding configurations, and synthetic evaluation artifacts.

### Key Root-Cause Findings:
1. **Holding Horizon Misalignment (180s vs 30s Alpha Half-Life):**
   Empirical analysis demonstrates that 10-second BTCUSDT directional alpha peaks at 20–30 seconds (2–3 bars) with a mean move of $\approx 0.74\text{ bps}$. However, the live evaluation loop (`feature engineering scripts/scratch/live_45m_comprehensive_eval.py`) configured `MAX_HOLD_BARS = 18` (180 seconds / 3 minutes), while the dashboard paper engine configured `maxHoldingBars = 30` (300 seconds / 5 minutes). Holding trades for 180s caused positions to outlive their alpha by $6\times$, colliding head-on with secular macro market drift (+25.18 bps upward trend over 45 min). All 5 authorized live signals suffered 100% time-stop expirations and net negative returns.
2. **Unreachable Static Barriers (16/12 bps vs 10s Volatility of 1–3 bps):**
   Fixed profit-take (+16 bps) and stop-loss (-12 bps) barriers were hardcoded into the live evaluation scripts. Over a 10s–30s horizon, normal BTCUSDT volatility generates moves of 0.5–2.0 bps. The +16 bps barrier was mathematically unattainable within short horizons absent extreme volatility spikes, rendering the take-profit barrier completely inert.
3. **Absence of Real Order-Book Queue Modeling & Maker Post-Only Execution:**
   - In `feature engineering scripts/inference/paper_trading_engine.py`, trades were modeled using continuous rebalancing against bar `close_price` with retail taker fees (10 bps) + slippage (4 bps), totaling 14 bps round-trip friction.
   - In `futures-prediction-dashboard/src/lib/paperEngine.ts`, "MAKER" execution was simulated by granting **instant, free fills** at `bestBid` (for buy) or `bestAsk` (for sell) without checking queue depth, order book queue priority, or trade flow exhaustion.
   - While a manual testnet script (`feature engineering scripts/scratch/test_binance_testnet_order.py`) demonstrated placing limit orders with `timeInForce: "GTX"` (Post-Only), **zero production or backtest loops incorporate queue position tracking, adverse selection modeling, or cancel/reprice mechanics**.
4. **Missing Institutional Verification Harnesses:**
   The verification scripts specified in `ORIGINAL_REQUEST.md`:
   - `scripts/evaluate_ensemble_and_meta.py`
   - `scripts/purged_walk_forward_backtest.py`
   - `tests/test_leakage_and_features.py`
   **do not currently exist in the repository**. Existing evaluation scripts (`rigorous_evaluation_audit.py`, `live_45m_comprehensive_eval.py`) are post-hoc diagnostic tools rather than automated walk-forward backtesters or leak-free unit test suites.
5. **Intra-Bar Lookahead Leakage in Sentiment Branch:**
   In `inference/checkpoint_loaders.py` (lines 115–120), `_live_micro_signal` samples `_live_ret_pct`, `_live_taker_imb`, and `_live_ofi_near` accumulated by the live WebSocket ingestor *during* the inference window. This injects unclosed bar return leakage into the signal, creating an artificial short-term hit rate boost that collapses when evaluated strictly out-of-sample.

---

## 2. In-Depth Codebase Investigation: Execution Logic & Order Placement

We audited every file in the repository that handles order execution, trade tracking, or fee calculations.

### 2.1 File-by-File Execution Logic Audit

| Component | Path | Execution Model | Order Types | Fees Modeled | Key Flaws |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Production Paper Engine** | `feature engineering scripts/inference/paper_trading_engine.py` | Continuous position rebalancing on bar `close_price` | None (Continuous signed fraction) | 14.0 bps RT (Taker fee 10 bps + Slippage 4 bps) | No discrete limit orders, no queue modeling, no holding horizon, no barriers. Destroys alpha with 14 bps taker friction. |
| **Live 45m Evaluation Script** | `feature engineering scripts/scratch/live_45m_comprehensive_eval.py` | Discrete trade tracking at BBO mid / close | None (Marked at BBO mid) | 10.0 bps RT (Taker fee) | Fixed 16 bps TP, 12 bps SL, 180s time expiry. 100% time-stop expirations. Held into secular drift. |
| **Scratch Paper Engine Test** | `feature engineering scripts/scratch/test_improved_paper_engine.py` | Event simulation on 181 live bars | Assumed Maker fill on TP/veto, Taker on SL | 3.6 bps RT Maker (with BNB), 6.3 bps RT Taker | Assumed instant fill at exact bar price for all exits. No queue modeling or adverse selection check. |
| **Dashboard Client Engine** | `futures-prediction-dashboard/src/lib/paperEngine.ts` | Discrete orders with Binance margin/brackets | "MAKER" vs "TAKER" toggle | VIP0: 4.0 bps RT Maker, 10.0 bps RT Taker | If role="MAKER", fills instantly at `bestBid`/`bestAsk`. Zero queue queue simulation, no queue wait time, zero adverse selection. |
| **Binance Testnet Order Script** | `feature engineering scripts/scratch/test_binance_testnet_order.py` | Isolated REST call to Binance Futures Testnet | `LIMIT` with `timeInForce: "GTX"` | N/A (Testnet API call) | Hardcoded test script placing order 15% below market and immediately deleting it. Not integrated into any execution loop. |

### 2.2 Order Types & Pure Maker-Post (GTX) Execution Analysis

Under Binance USDⓈ-M Futures fee schedules:
- **Taker Fee (VIP 0):** 0.05% per leg = **10.0 bps round-trip**.
- **Maker Fee (VIP 0):** 0.02% per leg = **4.0 bps round-trip**.
- **BNB 10% Discount:** Reduces maker fee to 0.018% per leg = **3.6 bps round-trip**.

Given that the empirical 10-second alpha move at peak ($H3$) is $+0.74\text{ bps}$ (mean gross edge of active signals), **taker execution is mathematically guaranteed to generate negative net PnL**:
$$\text{Net Taker PnL} = 0.74\text{ bps} - 10.0\text{ bps} = -9.26\text{ bps per trade}$$
Under maker execution, net friction is reduced by 6.0 bps to 4.0 bps RT. However, because gross alpha is 0.74 bps, even pure maker execution requires:
1. **A move-magnitude hurdle gate**: Only authorizing trades when predicted move $|\Delta \hat{p}| \ge 4.0\text{ bps} + \text{buffer}$.
2. **Vol-scaled barrier synchronization**: Sizing targets dynamically to realized volatility.
3. **Queue-Aware Fill Probability Modeling**: Mitigating adverse selection.

#### The Role of Post-Only (`GTX`):
On Binance Futures, placing a limit order with `timeInForce: "GTX"` guarantees that the order is rejected if it would cross the spread and execute as a taker. In `test_binance_testnet_order.py` (lines 63–70):
```python
order_params = {
    "symbol": "BTCUSDT",
    "side": "BUY",
    "type": "LIMIT",
    "timeInForce": "GTX",  # Post-Only Maker
    "quantity": "0.005",
    "price": str(test_price)
}
```
However, in neither `paper_trading_engine.py` nor `live_loop_v2.py` is `GTX` or limit order logic actually implemented. The system operates entirely on simulated instantaneous fills.

### 2.3 Queue Modeling & Adverse Selection (The Missing Layer)

In reality, placing a `GTX` limit order at the Best Bid (`bestBid`) does NOT guarantee execution:
1. **Queue Priority (FIFO):** At any given moment, there is already an existing queue of resting buy orders at `bestBid` with aggregate quantity $Q_{\text{ahead}}$. A newly submitted order enters at the *tail* of this queue.
2. **Fill Requirement:** The order will only fill if incoming aggressive taker sell flow exceeds $Q_{\text{ahead}}$:
   $$\sum \text{TakerSellVolume} \ge Q_{\text{ahead}} + Q_{\text{order}}$$
3. **Adverse Selection:** If price drifts upward, aggressive sell flow never arrives at `bestBid`; the order remains unfilled while the market rallies (failing to capture winning alpha). Conversely, if price drops rapidly, aggressive sells overwhelm `bestBid`, filling our order right before the price crashes through the bid level (adverse fill).
4. **Current Status in Codebase:** **Completely absent.** No queue tracking, no order book depth weighting, and no trade tape volume integration exist in any backtest or simulation engine.

### 2.4 Holding Period Flaw: Why Trades Were Held 180s

In `feature engineering scripts/scratch/live_45m_comprehensive_eval.py`:
- Line 46: `MAX_HOLD_BARS = 18` (18 bars $\times$ 10s = 180s / 3 minutes).
- Line 570: `hit_expiry = hold_bars >= MAX_HOLD_BARS`.

In `futures-prediction-dashboard/src/lib/paperEngine.ts`:
- Line 41: `private maxHoldingBars = 30; // 5.0 minutes`.

#### Why 180s Was Chosen:
Historical plan documentation (§10/§12) assumed that ensemble sequence models (TFT, PatchTST, SSM) trained on longer horizon labels (`label_return_h1` = 12 bars / 2m, `label_return_h2` = 60 bars / 10m) required 3 to 5 minutes to express directional momentum.

#### The Empirical Reality:
Empirical inspection of [`scratch/live_45m_eval_ledger.jsonl`](file:///D:/backup/Binance%20data/scratch/live_45m_eval_ledger.jsonl) demonstrates that directional information decays rapidly:
- **$H1$ (10s):** Raw edge $= +0.245\text{ bps}$
- **$H2$ (20s):** Raw edge $= +0.236\text{ bps}$
- **$H3$ (30s):** Raw edge $= +0.227\text{ bps}$ (Peak win rate on decisive bars: 59.1%)
- **$H6$ (60s):** Raw edge $= -0.125\text{ bps}$
- **$H12$ (120s):** Raw edge $= -0.562\text{ bps}$
- **$H18$ (180s):** Raw edge $= +0.034\text{ bps}$

Holding trades for 180s forced trades to remain open long after the 30s alpha window had dissolved. Because BTC drifted up by $+25.18\text{ bps}$ over the 45-minute audit window ($+1.68\text{ bps}$ drag per 180s window), holding SHORT positions for 180s caused 100% time-stop expirations with severe losses.

### 2.5 Barrier Logic: Static 16/12 bps vs Dynamic ATR

In `live_45m_comprehensive_eval.py`:
```python
TP_BPS = 16.0  # +16 bps take profit
SL_BPS = 12.0  # -12 bps stop loss
```
Over a 30s window (3 10-second bars), 1-standard-deviation BTC price movement is:
$$\sigma_{30s} \approx \sqrt{3} \times \sigma_{10s} \approx 1.73 \times 0.74\text{ bps} \approx 1.28\text{ bps}$$
A static $+16.0\text{ bps}$ target represents an excursion of **$> 12$ standard deviations** over 30s. Consequently:
- Take-profit hit rate was **0.0%**.
- Stop-loss hit rate was **0.0%**.
- 100% of trades resolved via `TIME_EXPIRY`.

#### Required Solution:
Barriers must be scaled dynamically to realized volatility ($\text{ATR}_{10s}$ or Parkinson volatility):
$$\text{TP} = k_{\text{TP}} \times \text{ATR}_{10s}, \quad \text{SL} = k_{\text{SL}} \times \text{ATR}_{10s}$$
For example, with $k_{\text{TP}} = 1.5$ and $\text{ATR}_{10s} \approx 1.2\text{ bps}$, the dynamic barrier is $\approx 1.8\text{ bps}$, synchronizing directly with the observable 3-bar excursion.

### 2.6 Cancellation and Repricing Logic

In a production maker node, resting limit orders must be actively managed:
1. **Adverse Queue Movement:** If book imbalance deteriorates (e.g. bid depth plummets while ask depth surges), cancel immediately.
2. **Price Level Shift:** If the Best Bid moves up (market moves away), cancel and reprice at the new Best Bid, provided the model still authorizes the entry.
3. **Stale Signal Timeout:** If an order sits unfilled for $> 2$ bars (20s), cancel to avoid stale fills.
4. **Current Status in Codebase:** No cancellation or repricing logic currently exists in `live_loop.py`, `live_loop_v2.py`, or `paper_trading_engine.py`.

---

## 3. Existing Backtest & Evaluation Scripts: Audit & Gap Analysis

We inspected all existing evaluation and validation scripts across `feature engineering scripts/` and `scratch/`.

### 3.1 Review of Existing Scripts

#### 1. `feature engineering scripts/scratch/rigorous_evaluation_audit.py`
- **Status:** **Operational & Mathematically Sound.**
- **Functionality:** Audits `scratch/live_45m_eval_ledger.jsonl`. Backfills BBO mid-prices, computes forward returns for $H1, H2, H3, H6, H12, H18$, computes Wilson 95% CIs, exact binomial p-values, clusters overlapping signals into 18 independent runs, computes drift-adjusted returns, baselines (Always-Long, Always-Short, Random MC, Best Member, Climatological Brier), and Expected Calibration Error (ECE).
- **Limitations:** Only audits the static 45-minute live telemetry ledger. It is not an automated backtest engine and does not simulate walk-forward order placement or queue dynamics.

#### 2. `feature engineering scripts/validation/regime_holdout_eval.py`
- **Status:** **Operational but limited by available OOF data.**
- **Functionality:** Compares out-of-fold member predictions against naive majority-class baselines across 3 validation slices to apply kill-criteria (§10.3).
- **Limitations:** Only checks directional classification accuracy; does not evaluate PnL, fees, execution mechanics, or trade duration.

#### 3. `feature engineering scripts/scratch/test_improved_paper_engine.py`
- **Status:** **Incomplete Prototype.**
- **Functionality:** Runs a heuristic grid search over 181 bars from `monitor_30m_log.jsonl` testing various TP/SL and trailing profit locks.
- **Limitations:** Assumes fills occur at bar prices with zero queue modeling; highly overfitted to 181 specific bars.

#### 4. `feature engineering scripts/tests/wf_check.py`
- **Status:** **Sanity script.**
- **Functionality:** Computes purged train/test split on a single day's parquet file for LightGBM.
- **Limitations:** Hardcoded to single day, does not simulate maker execution or multi-horizon ensembles.

### 3.2 Missing Verification Harnesses (Must Be Built)

| Required Script | Specified In | Purpose | Status | Required Work |
| :--- | :--- | :--- | :--- | :--- |
| `scripts/evaluate_ensemble_and_meta.py` | `ORIGINAL_REQUEST.md` (Verification #2) | Evaluates ensemble member de-correlation (AC1: $r \le 0.70$), meta-labeler ECE (AC2: $\le 8.0\%$), and Brier score ($< 0.2786$). | **MISSING** | Create production script loading validation OOF data, computing Pearson/Spearman matrices, evaluating multi-class calibration, and generating audit reports. |
| `scripts/purged_walk_forward_backtest.py` | `ORIGINAL_REQUEST.md` (Verification #3) | Runs purged walk-forward cross-validation backtest with BBO queue maker execution, realistic 4 bps maker fees, dynamic ATR barriers, and $\ge 200$ independent clusters (AC4). | **MISSING** | Create complete discrete-event backtesting engine with BBO queue tracking, GTX post-only execution, purged walk-forward fold looping, and cluster-level performance metrics. |
| `tests/test_leakage_and_features.py` | `ORIGINAL_REQUEST.md` (Verification #1) | Pytest suite validating zero feature leakage ($\le T_{\text{close}} - 1000\text{ms}$), sentiment branch cleanliness, and inference latency $\le 250\text{ms}$ (AC3, AC5). | **MISSING** | Create end-to-end pytest test harness covering AC1–AC5 acceptance criteria. |

---

## 4. Purged Walk-Forward Cross-Validation Architecture

### 4.1 Available Historical Dataset on Disk

An exhaustive scan of `D:\backup\Binance data` identified the following usable datasets:
1. **7-Day Out-of-Fold (OOF) Prediction Parquets (`feature engineering scripts/fusion/oof/`):**
   - `gbm-family.parquet`: **60,480 bars** (2026-10-04 23:00 to 2026-10-11 22:59)
   - `sentiment-branch.parquet`: **60,480 bars** (2026-10-04 23:00 to 2026-10-11 22:59)
   - `regime_states.parquet`: **60,480 bars** (2026-10-04 23:00 to 2026-10-11 22:59)
   - `itransformer.parquet`: **6,040 bars** (strided by 10 bars / 100s)
   - `nhits.parquet`: **6,040 bars**
   - `patchtst.parquet`: **6,040 bars**
   - `tft.parquet`: **6,040 bars**
   - `ssm-challenger.parquet`: **6,012 bars**
   - `kronos-finetune.parquet`: **1,249 bars** (strided by 48 bars)
2. **Feature-Engineered 10s Parquet:**
   - `feature engineering scripts/output/features_BTCUSDT_2026-10-11.parquet`: **8,640 bars** (1 full 24-hour day of 10-second bars with 226 columns, including BBO bid/ask, CVD, order flow imbalance, OHLCV).
3. **Live Evaluation Event Ledgers (`scratch/`):**
   - `scratch/live_45m_eval_ledger.jsonl`: **270 bars** with all member predictions, BBO prices, and backfilled forward returns.
   - `scratch/live_30m_eval_ledger.jsonl`: **193 bars**.

### 4.2 Structuring Purged Walk-Forward CV

To satisfy AC4 ($\ge 200$ independent signal clusters across multiple volatility regimes without lookahead leakage):

#### Mathematical Purge & Embargo Rules:
1. **Prediction Horizon:** $H = 3\text{ bars}$ ($30\text{ seconds}$).
2. **Purge Window:** When validating on window $[t_{\text{val\_start}}, t_{\text{val\_end}})$, all training samples within $[t_{\text{val\_start}} - H, t_{\text{val\_start}})$ must be **purged** (discarded from training) because their 3-bar forward outcome overlaps with the validation window.
3. **Embargo Window:** An embargo period of $E \ge 3\text{ bars}$ must be applied immediately following the validation window before subsequent training data can be used, preventing auto-regressive feature leakage.
4. **Independent Signal Clustering:** Consecutive same-direction signals within $H$ bars must be clustered into a single independent run. Only the first bar of each run is evaluated to guarantee independent Bernoulli trials.

#### Fold Configuration Across the 7-Day Dataset:
With 60,480 bars (7 days):
- **Expanding Window Splits:**
  - Fold 1: Train Day 1–2 (17,280 bars) $\to$ Purge 3 bars $\to$ Val Day 3 (8,640 bars) $\to$ Embargo 3 bars
  - Fold 2: Train Day 1–3 (25,920 bars) $\to$ Purge 3 bars $\to$ Val Day 4 (8,640 bars) $\to$ Embargo 3 bars
  - Fold 3: Train Day 1–4 (34,560 bars) $\to$ Purge 3 bars $\to$ Val Day 5 (8,640 bars) $\to$ Embargo 3 bars
  - Fold 4: Train Day 1–5 (43,200 bars) $\to$ Purge 3 bars $\to$ Val Day 6 (8,640 bars) $\to$ Embargo 3 bars
  - Fold 5: Train Day 1–6 (51,840 bars) $\to$ Purge 3 bars $\to$ Val Day 7 (8,640 bars)
- Across the 5 out-of-sample validation days (43,200 bars), a typical 1%–2% signal generation rate yields **$400\text{ to }800$ independent signal clusters**, comfortably exceeding the $\ge 200$ cluster threshold required by AC4.

---

## 5. Test Infrastructure & Acceptance Criteria Verification

### 5.1 Existing Unit Test Review (`feature engineering scripts/tests/`)
- `test_offline.py`: Validates news scraping, regex matching, and offline pipeline components. No execution or leakage tests.
- `test_live_node_e2e.py`: Exercises `LiveStreamingPredictionNode` with 100 synthetic bars. It validates component connectivity, but lacks pytest assertions for AC1–AC5 criteria.
- `test_news_pipeline.py`, `test_gapfix.py`, `test_streaming_offline.py`: Module-specific unit tests.

### 5.2 Test Blueprint for `tests/test_leakage_and_features.py`
The new test suite must be placed in `tests/test_leakage_and_features.py` (at workspace root) and execute via `pytest tests/test_leakage_and_features.py -v`. It must test:

1. **Test 1: Strict Timestamp Latency Offset ($\ge 1000\text{ms}$)**
   - Verify that for any bar closing at $T_{\text{close}}$, all feature inputs are timestamped $\le T_{\text{close}} - 1000\text{ms}$.
   - Assert zero access to trade tape or order book ticks occurring in the $(T_{\text{close}} - 1000\text{ms}, T_{\text{close}}]$ window.
2. **Test 2: Micro-Signal / Sentiment Branch Zero-Leakage Audit**
   - Mock price series with artificial future price spikes. Verify that `_live_micro_signal` and `sentiment-branch` predictions do not change when future price data is perturbed.
   - Assert `_live_ret_pct` is strictly removed or derived exclusively from closed historical bars.
3. **Test 3: Feature Contract Stationary Verification**
   - Verify all 174 features conform to `FEATURE_COLUMNS` in `common/feature_contract.py`.
   - Assert zero NaN, zero Inf, and no lookahead operators (`lead`, unshifted diffs).
4. **Test 4: Inference Latency Benchmark ($\le 250\text{ms}$)**
   - Benchmark end-to-end forward pass on CPU/GPU. Assert inference turnaround $\le 250\text{ms}$ (meeting the 10-second bar safety budget).
5. **Test 5: Execution Logic & GTX Post-Only Rule Validation**
   - Test limit order placement at BBO.
   - Verify maker fee rate applied is 2.0 bps (4.0 bps RT).
   - Verify order rejection when crossing the spread (`timeInForce="GTX"`).
   - Test FIFO queue exhaustion logic: verify order does not fill until cumulative opposite trade volume exceeds queue ahead.

---

## 6. Target Execution & Backtest Architecture (Blueprint for Implementation)

To enable seamless implementation by the Execution Engine Specialist and Quant ML Engineer, the proposed architecture is structured into three clean, modular components:

### 6.1 `scripts/purged_walk_forward_backtest.py`
```
Architecture:
├── PurgedWalkForwardEngine
│   ├── make_purged_folds(df, n_folds=5, embargo_bars=3, purge_bars=3)
│   ├── BBOQueueSimulator (Queue position, volume depletion, adverse selection)
│   ├── DynamicBarrierManager (ATR_10s scaling, 3-bar holding timeout)
│   └── ClusterMetricsReporter (Wilson 95% CI, binomial p-val, net PnL after 4 bps maker fee)
```

#### Queue Simulation Model:
For each bar $t$ where a signal is emitted:
1. Long Signal $\to$ Place `GTX` BUY limit order at $P_{\text{bid}}(t)$.
2. Queue depth ahead: $Q_{\text{ahead}} = \text{bids\_depth}(t) \times \gamma$ (where $\gamma \approx 0.5$ represents random arrival position).
3. Over the next bar $t+1$:
   - Check taker sell volume: $V_{\text{sell}} = \text{sell\_volume}(t+1)$.
   - If $V_{\text{sell}} \ge Q_{\text{ahead}} + Q_{\text{order}}$, order is FILLED at $P_{\text{bid}}(t)$ as MAKER (fee: 2.0 bps).
   - If $P_{\text{ask}}(t+1) \le P_{\text{bid}}(t)$, price swept through $\to$ FILLED with adverse selection.
   - If $P_{\text{bid}}(t+1) > P_{\text{bid}}(t)$ and $V_{\text{sell}} < Q_{\text{ahead}}$, order is UNFILLED (price moved away) $\to$ CANCEL order.
4. If filled, evaluate dynamic barriers over subsequent 3 bars ($H3$):
   - $\text{TP} = P_{\text{fill}} + k_{\text{TP}} \times \text{ATR}_{10s}$
   - $\text{SL} = P_{\text{fill}} - k_{\text{SL}} \times \text{ATR}_{10s}$
   - Hard exit at $t_{\text{fill}} + 3\text{ bars}$ ($30\text{ seconds}$).

### 6.2 `scripts/evaluate_ensemble_and_meta.py`
```
Architecture:
├── EnsembleEvaluator
│   ├── compute_correlation_matrix(oof_predictions) -> Pearson & Spearman
│   ├── verify_feature_families(feature_df) -> Assert >= 3 distinct families
│   ├── evaluate_meta_labeler(meta_filter, val_df) -> Continuous in [0, 1]
│   ├── compute_ece_and_reliability(probs, y_true, n_bins=10)
│   └── compare_brier_score(probs, y_true, climatological_prior)
```

### 6.3 `tests/test_leakage_and_features.py`
Standard Pytest test file located at `tests/test_leakage_and_features.py` executable directly via:
```powershell
pytest tests/test_leakage_and_features.py -v
```

---

## 7. Summary & Recommendations for Team Orchestrator

1. **Immediate Execution Engine Overhaul:** Replace the simulated continuous rebalancing engine in `paper_trading_engine.py` and the naive dashboard engine with a BBO queue-aware discrete maker node.
2. **Synchronize Barriers & Horizon:** Enforce 3-bar ($30\text{s}$) maximum hold time across all execution paths. Replace static 16/12 bps targets with dynamic $k \times \text{ATR}_{10s}$ volatility scaling.
3. **Purge Lookahead Leakage:** Remove intra-bar unclosed returns (`_live_ret_pct`) from `inference/checkpoint_loaders.py` and enforce the 1000ms latency cutoff.
4. **Implement Missing Scripts:** Prioritize authoring `tests/test_leakage_and_features.py`, `scripts/evaluate_ensemble_and_meta.py`, and `scripts/purged_walk_forward_backtest.py` as the formal gatekeepers for AC1–AC5.
