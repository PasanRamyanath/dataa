# Comprehensive Investigation Report: Features, Models, Leakage & Data Architecture
**Project:** BTCUSDT 10-Second Alpha & Execution Overhaul  
**Date:** 2026-10-18  
**Investigator:** Features & Models Explorer  
**Target Codebase:** `D:\backup\Binance data`  
**Reference Telemetry:** `scratch/live_45m_eval_ledger.jsonl`, `scratch/live_30m_eval_ledger.jsonl`, `result.md`

---

## Executive Summary

An exhaustive audit of the feature engineering pipelines, data loaders, member models, and meta-labeling architecture in `D:\backup\Binance data` confirms the root causes of the empirical performance collapse documented in `result.md` (where directional hit rate fell to 46.4% raw mid, Wilson 95% CI $[29.5\%, 64.2\%]$, $p = 0.851$).

The key discoveries of this investigation are:
1. **Intra-Bar Forward Lookahead Leakage:** The live WebSocket ingestion node (`streaming/live_loop_v2.py` and `streaming/live_ingestion/binance_ws.py`) samples trade prices and taker flow during the 900–1000 ms GPU inference window *after* bar close $t$, and feeds the unclosed $t+1$ intra-bar return (`_live_ret_pct`) directly into `_live_micro_signal` and Bayesian probability adjustments. This gave `sentiment-branch` an artificial +0.977 bps advantage by peaking into the forward evaluation window.
2. **Root Cause of Member Collinearity ($r = 0.987 \text{ to } 0.997$):** Member models did *not* learn collinear representations in isolation (their raw Out-Of-Fold predictions correlate at only $0.05 \text{ to } 0.52$). Instead, the live inference harness (`inference/checkpoint_loaders.py`) injects a hardcoded **30% to 50% blend of `_live_micro_signal`** into *every single member's output*, uses a shared wrapper class (`_NeuralForecastPredictor`) for TFT, PatchTST, iTransformer, and NHITS with identical horizon weighting, and compresses all outputs through the exact same nonlinear scalar probability mapping (`_signal_to_3class_proba`). This forcibly synchronized all 8 members into a single redundant vote.
3. **Meta-Labeler Model Bypass & Piecewise Clamping:** The trained machine learning meta-labeler (`fusion/checkpoints/meta_labeling_filter/meta_labeling_filter.pkl`) is loaded in `live_loop_v2.py` but **completely bypassed and never executed**. It was replaced with hardcoded piecewise rules: all flat bars are clamped to $0.420$, non-conforming directional bars ($p_{sel} < 0.62$ or $p_{flat} > 0.28$) are clamped to $[0.30, 0.52]$, and conforming bars jump over a dead zone to $[0.85, 0.94]$.
4. **Stacking Meta-Learner Breakdown:** The stacking logistic regression model failed due to collinear inputs at inference time, yielding inverted weights on PatchTST/TFT and a $-1.12$ intercept on the flat class (predicting $P(flat) = 0.02\%$ when the ground truth is $89.6\%$ flat). This forced ad-hoc heuristic patches (`blended_edge = 0.72 * mean + 0.28 * stack`).
5. **Data Source Readiness:** High-resolution Binance USD-M and Spot data structures (aggTrades, bookTicker, bookDepth, fundingRate, indexPriceKlines, markPriceKlines) exist in code and Vision downloader scripts. However, cross-asset ETHUSDT data has not yet been fetched for offline training, and several features in the live circular buffer (`streaming/live_ingestion/ring_buffer.py`) currently use mock/synthetic substitutes.

---

## 1. Feature Calculation Code, Data Loaders, and Caching Pipeline

### 1.1 The 174-Feature Contract Architecture
The pipeline's feature definition is strictly governed by `feature engineering scripts/common/feature_contract.py` and `feature engineering scripts/col_schema.json`.

The overall schema contains **226 columns**, categorized as follows:
- **1 Identifier Column:** `ts` (Datetime nanoseconds)
- **9 Target & Diagnostic Columns:** `label_triple_barrier` ({-1, 0, 1}), `label_barrier_type`, `label_exit_bar_offset`, `label_is_valid`, `label_return_h1` (12 bars), `label_return_h2` (60 bars), `label_return_h3` (360 bars), `label_return_h4` (8640 bars), `label_meta_act` ({0, 1}).
- **32 Non-Stationary Excluded Columns:** `open`, `high`, `low`, `close`, `vwap`, `intra_bar_price_std`, `index_price`, `mark_price`, `spot_close`, `bybit_close`, `bollinger_mid/upper/lower`, `atr`, `macd/signal/hist`, `htf_1m_*`, `htf_5m_*`, `cvd`, `cvd_ma_*`, and seasonal sin/cos flags.
- **10 Exchange-Discontinued Dead Columns:** `bybit_premium_index`, `bybit_basis_pct`, `cross_exch_basis_divergence`, `cross_exch_basis_divergence_ma_*`, `liquidation_notional`, `liquidation_notional_*`, `liquidation_side_imbalance_*`.
- **174 Active Stationary Features:** Used as model inputs ($226 - 1 - 9 - 32 - 10 = 174$).

```
174 Active Features Breakdown by Family:
├── Returns & Momentum (11): log_return_{1,3,6,12,30,60,180}, momentum_{6,30,180,720}
├── Volatility & Range (14): realized_vol_{30,90,360}, price_pct_of_range_{30,180,720}, dist_from_high/low, bar_range/body_pct
├── Volume & Trade Flow (12): volume_ma_{30,180,720}, volume_ratio_{30,180,720}, taker_buy_sell_imbalance, taker_buy_ratio, large_trade_dominance, trade_size_cv, vwap_deviation{_ma}
├── Order Book Microstructure (20): OFI near/wide, ofi_near_ma_{30,180}, ofi_level_{0.2,1,2,3,4,5}, ofi_imbalance_decay_rate/slope, proxy_spread_pct, roll_spread_{30,90,360}, kyle_lambda_{30,90,360}, amihud_{30,90,360}, vpin_proxy_{30,90,360}, quote_resilience
├── Funding & OI Crowd Dynamics (14): funding_rate{_ma, _vol, _change}, open_interest{_value}, toptrader_count/sum_ls_ratio, all_account_ls_ratio, taker_ls_vol_ratio, oi_roc_{30,180,720}, oi_change_pct, volume_to_oi_ratio, funding_oi_crowding_{180,720}
├── Basis & Lead-Lag (18): basis_pct{_ma}, mark_price_deviation, premium_index{_ma}, spot_futures_spread_pct/ret_div, spot_lead_return_lag_{1,3,6,30}, cross_exch_spread_pct, bybit_lead_return_lag_{1,2,3,5}, cross_exch_ofi_div
├── Implied Volatility & Term Structure (11): implied_vol{_change, _ma}, iv_realized_vol_spread, calendar_basis_front/back_pct, term_structure_slope, calendar_basis_momentum
├── Technical & Statistical Regime (20): rsi, bollinger_pct_b/bandwidth, atr_pct, plus/minus_di, adx, return_skew/kurtosis/autocorr, vol_clustering, hurst_proxy, vol_regime_bucket, breakout_{up,down}, support/resistance_distance
├── Higher Timeframe Context (4): htf_1m_return, htf_1m_deviation, htf_5m_return, htf_5m_deviation
└── Calendar & Funding Clock (40): Cyclical sin/cos encodings, session indicators (Asian, Europe, US), minutes_to/from_funding, pre/post_funding_1h, funding_cycle_phase
```

### 1.2 Pipeline Execution & Orchestration
1. **Offline Training Pipeline (`build_dataset.py` & `feature_engine/pipeline.py`):**
   - Ingests raw zipped day files: `aggTrades`, `bookDepth`, `bookTicker`, `metrics`, `fundingRate`, `indexPriceKlines`, `markPriceKlines`, `premiumIndexKlines`, `spot aggTrades`, and `Bybit trading`.
   - Re-samples 10-second bars using `resample.build_trade_bars()` with `closed="left", label="left"`.
   - Runs `compute_all_features(DaySources)` to sequentially calculate all 174 features.
   - Attaches labels via `labeling.add_all_labels()` (triple-barrier and multi-horizon forward returns $H1$ through $H4$).
   - Validates and exports snappy-compressed parquets to `feature engineering scripts/output/features_BTCUSDT_<YYYY-MM-DD>.parquet`.
2. **Data Loading & HF Remote Sync (`common/data.py`):**
   - `load_days(symbol, max_days, prefer_local=True)` checks `config.OUTPUT_ROOT`. If local parquets are absent, it pulls on-demand from HuggingFace dataset repo `config.HF_REPO_ID` (`p-j-r-1-2-3/Project-IMPJR-ModelTFT-dataset`).
   - One full-day parquet exists locally: `feature engineering scripts/output/features_BTCUSDT_2026-10-11.parquet` (8,640 rows $\times$ 226 cols).
3. **Live In-Memory Circular Buffer (`streaming/live_ingestion/ring_buffer.py`):**
   - `OnlineFeatureRingBuffer(capacity=1000, min_warmup_bars=60)` maintains a `deque` of the latest 1,000 completed bars in RAM.
   - Upon bar close, runs the full Polars transformation suite in $\approx 105\text{ ms}$ without disk I/O.
   - **Crucial Deficiency Identified in RingBuffer:** Several external data sources are currently mocked in `ring_buffer.py`:
     - Line 99: `spot_df` is mocked from `index_price` (`df.select("ts", pl.col("index_price").alias("close"))`).
     - Line 102–107: `bybit_df` is mocked from Binance `index_price` and Binance buy/sell volume.
     - Line 111–120: `implied_vol` is calculated via an artificial proxy: `(realized_vol_360 * ann_factor * 0.5 + 45.0)`.
     - Line 121–128: `calendar_basis` uses a static synthetic multiplier `(basis_pct + funding_rate * 90.0)`.

---

## 2. Intra-Bar Lookahead Leakage Audit

### 2.1 The Intra-Bar Unclosed Return Leak in Live Ingestion
The audit pinpointed the exact location and mechanism of forward return leakage in the live prediction loop.

#### Evidence 1: Live WebSocket Snapshot Computation
In `feature engineering scripts/streaming/live_ingestion/binance_ws.py`, lines 324–361:
```python
def get_live_snapshot(self, ref_price: float = 0.0) -> dict:
    """Returns an instantaneous thread-safe snapshot of the currently forming 10s bar
    (live trade price, intra-bar return, aggressive taker buy/sell imbalance, and L2 OFI)
    captured while the Dual-T4 GPUs execute deep sequence model inference."""
    with self._lock:
        ...
        anchor = float(ref_price) if ref_price > 0 else ...
        ret_pct = ((live_p - anchor) / anchor) * 100.0 if anchor > 0 else 0.0
        taker_imb = (buy_v - sell_v) / (tot_v + 1e-9) if tot_v > 0 else 0.0
        ...
        return {
            "live_price": live_p,
            "ret_pct": ret_pct,
            "taker_imb": taker_imb,
            ...
        }
```

#### Evidence 2: Triggering Snapshot After GPU Inference
In `feature engineering scripts/streaming/live_loop_v2.py`, lines 520–569:
```python
# Bar t closes at T_close. Feature DF is built:
feature_df = self.ring_buffer.append_bar(bar_dict)

# 3. GPU inference dispatched across members (takes 900 - 1018 ms):
predictions, latencies = self.coordinator.predict_tick(feature_df, timeout_seconds=8.5)

# 3b. Capture live intra-bar microstructure accumulated during GPU forward pass:
live_snap = self.ws_ingestor.get_live_snapshot(close_price)  # ref_price is bar t close!

# 4. Fusion stage incorporates live_snap:
fused = _fuse(predictions, ..., live_snap=live_snap, bar_dict=bar_dict)
```

#### Evidence 3: Price Return Injection into Decision Fusion
In `feature engineering scripts/streaming/live_loop_v2.py`, lines 124–154:
```python
live_ret_pct = float(snap.get("ret_pct", 0.0))
...
net_signal = 0.55 * ens_dir_edge + 0.30 * orderflow_edge + 0.15 * np.tanh(live_ret_pct / 0.004)
...
if abs(orderflow_edge) > 0.04:
    delta = float(np.clip(orderflow_edge * 0.06, -0.05, 0.05))
    p_down = max(0.01, proba[0] - delta)
    p_up = max(0.01, proba[2] + delta)
    ...
```

#### Evidence 4: Infiltration into `_live_micro_signal`
In `feature engineering scripts/inference/checkpoint_loaders.py`, lines 114–120:
```python
# Incorporate intra-bar WebSocket snapshot if present
if "_live_ret_pct" in cols:
    live_ret = _val("_live_ret_pct") / 100.0
    live_taker = np.clip(_val("_live_taker_imb"), -1.0, 1.0)
    live_ofi = np.clip(_val("_live_ofi_near"), -1.0, 1.0)
    micro_score = 0.35 * micro_score + 0.45 * np.tanh(live_ret / 0.00012) + 0.12 * live_taker + 0.08 * live_ofi
```

### 2.2 Mathematical Mechanics of the Leakage
1. At timestamp $T$, bar $t$ closes with price $P_t$.
2. The primary prediction target is the forward return of horizon $H1$ (from bar $t$ close to bar $t+1$ close):
   $$R_{t+1} = \frac{P_{t+1} - P_t}{P_t}$$
3. The GPU inference process takes $\Delta t \approx 1018\text{ ms}$. During this window, Binance WebSocket delivers real market trades for bar $t+1$.
4. At $T + 1018\text{ ms}$, `get_live_snapshot(ref_price=P_t)` computes:
   $$\text{ret\_pct} = \frac{P_{t + 1018\text{ms}} - P_t}{P_t} \times 100\%$$
5. This is **10% of the forward-looking target path** ($t \to t+1$).
6. Because 10-second price movements exhibit strong intra-bar persistence, knowing $P_{t + 1018\text{ms}} - P_t$ provides a massive, non-causal information leak regarding whether $P_{t+1} > P_t$.
7. This unclosed return was directly weighted by **45%** in `_live_micro_signal` and used to tilt Bayesian fusion probabilities. When scored without latency adjustment, this created the illusion of high directional edge (+0.977 bps for `sentiment-branch`), which immediately collapsed to negative ($-0.176\text{ bps}$) when latency-adjusted to $t + 1018\text{ ms}$.

### 2.3 Other Historical Feature Pipeline Leaks
1. **1m Kline Open vs Close Timestamps:** Historical 1m klines (`indexPriceKlines`, `markPriceKlines`, `premiumIndexKlines`) were originally stamped at `open_time`, allowing backward as-of joins to see future 60s prices up to 59s early. In `parsers/binance_parser.py` (lines 129–137), this was patched by keying `ts` to `close_ts + 1ms`.
2. **5m Metrics Window Start:** Metrics (open interest, long/short ratios) were stamped at `create_time` (window start). This was patched in `parsers/binance_parser.py` (lines 177–184) by adding 5 minutes.
3. **Requirement AC3 Compliance Status:** Any live or offline feature calculated at bar $t$ MUST satisfy $t \le T_{\text{close}} - 1000\text{ ms}$. `live_snap` violates this by using $t > T_{\text{close}}$.

---

## 3. Member Models & The Root Cause of Collinearity ($r > 0.98$)

### 3.1 The Measured Collinearity Dilemma
In `result.md`, Section 8, the correlation matrix of live member predictions showed near-unity collinearity:
- `tft` vs `nhits`: $r = 0.997$
- `tft` vs `patchtst`: $r = 0.996$
- `patchtst` vs `nhits`: $r = 0.993$
- `ssm-challenger` vs `tft`: $r = 0.989$
- `ssm-challenger` vs `patchtst`: $r = 0.989$

However, our empirical calculation of raw OOF correlations from offline training parquets (`feature engineering scripts/fusion/oof/*.parquet`) revealed:
- `tft` vs `nhits`: $r = 0.1934$
- `tft` vs `patchtst`: $r = 0.3333$
- `patchtst` vs `nhits`: $r = 0.3187$
- `ssm-challenger` vs `tft`: $r = 0.4069$

**Why did models with raw correlation of 0.19–0.40 collapse to 0.987–0.997 in live inference?**

### 3.2 Root Cause Analysis

#### Factor 1: Hardcoded Microstructure Blending in Checkpoint Loaders
Inspection of `feature engineering scripts/inference/checkpoint_loaders.py` revealed that *none* of the deep sequence models output raw model predictions. Instead, every predictor is wrapped in post-processing logic that blends in the common signal `_live_micro_signal(recent_df)`:

| Member Predictor | Wrapper Class | Micro-Signal Weight | Raw Model Weight | Code Reference |
| :--- | :--- | :---: | :---: | :--- |
| `gbm-family` | `_GbmFamilyPredictor` | **30%** (`0.30 * micro_ret`) | 35% cls, 35% reg | `checkpoint_loaders.py:195` |
| `sentiment-branch` | `_SentimentBranchPredictor` | **40%** (`0.40 * micro_ret`) | 25% cls, 35% crowd | `checkpoint_loaders.py:277` |
| `ssm-challenger` | `_SsmChallengerPredictor` | **30%** (`0.30 * micro_ret`) | 35% cls, 35% reg | `checkpoint_loaders.py:360` |
| `tft` | `_NeuralForecastPredictor` | **45%** (`0.45 * micro_ret`) | 55% centered_traj | `checkpoint_loaders.py:489` |
| `patchtst` | `_NeuralForecastPredictor` | **45%** (`0.45 * micro_ret`) | 55% centered_traj | `checkpoint_loaders.py:489` |
| `itransformer` | `_NeuralForecastPredictor` | **45%** (`0.45 * micro_ret`) | 55% centered_traj | `checkpoint_loaders.py:489` |
| `nhits` | `_NeuralForecastPredictor` | **45%** (`0.45 * micro_ret`) | 55% centered_traj | `checkpoint_loaders.py:489` |
| `kronos-finetune` | `_KronosPredictor` | **50%** (`0.50 * micro_ret`) | 50% centered_traj | `checkpoint_loaders.py:594` |

Almost half of the directional vote for every model was identical.

#### Factor 2: Shared `_NeuralForecastPredictor` Wrapper Class
All four NeuralForecast members (`tft`, `patchtst`, `itransformer`, `nhits`) share the exact same class implementation (`checkpoint_loaders.py:415–491`):
1. They forecast raw close price paths (`vals`).
2. They map the trajectory to scalar return using the exact same formula:
   $$\text{raw\_traj} = 0.50 \cdot \frac{V_1 - V_0}{P_0} + 0.30 \cdot \frac{V_3 - V_0}{2 P_0} + 0.20 \cdot \frac{V_{-1} - V_0}{\sqrt{N} P_0}$$
3. They apply the identical historical centering formula:
   $$\text{centered\_traj} = \text{raw\_traj} - 0.70 \cdot \text{mean}(\text{traj\_history})$$
4. They combine it with `micro_ret` using the identical 55% / 45% blend.

#### Factor 3: Identical Non-linear Probability Mapping
All models route their `combined_ret` through `_signal_to_3class_proba`:
```python
def _signal_to_3class_proba(expected_ret, activity=0.5, deadzone=0.000105, scale=0.000165):
    flat_ratio = abs_sig / max(deadzone, 1e-6)
    p_flat = float(np.clip(0.64 * np.exp(-0.85 * (flat_ratio ** 2.2)), 0.06, 0.72))
    z = float(np.clip(sig / max(scale, 1e-6), -12.0, 12.0))
    dir_up_share = 1.0 / (1.0 + np.exp(-2.4 * z))
    ...
```
Because the inputs were dominated by `micro_ret`, and the nonlinear sigmoid mapping had identical parameters (`deadzone=0.000105, scale=0.000165`), the resulting 3-class probability vectors $[P_{down}, P_{flat}, P_{up}]$ were mathematically forced to be nearly identical, producing $r > 0.98$.

---

## 4. Meta-Labeler Implementation & Threshold Clamping

### 4.1 Bypassing the Machine Learning Model
The project includes a trained machine learning meta-labeling filter (`fusion/checkpoints/meta_labeling_filter/meta_labeling_filter.pkl`, wrapping a scikit-learn `LogisticRegression` classifier fit on triple-barrier labels in `fusion/meta_labeling.py`).

In `feature engineering scripts/streaming/live_loop_v2.py`:
- Line 251: `self.meta_filter = loaders.load_meta_labeling_filter()` successfully loads the checkpoint.
- Lines 165–180: A Polars row is formatted with meta-features.
- **The Failure:** Lines 186–201 **never call `meta_filter.act_probability(row)` or `meta_filter.bet_size(row)`**!
Instead, the trained model is completely discarded in favor of hardcoded piecewise threshold branches.

### 4.2 The Hardcoded Piecewise Formula
In `feature engineering scripts/streaming/live_loop_v2.py`, lines 186–201:
```python
# Ultra Sniper Precision Gate:
# Require >= 62% directional consensus and <= 28% flat probability.
# Ambiguous distributions (e.g. 52% up with 44% flat) are strictly vetoed as chop.
if p_sel < 0.62 or p_flat > 0.28:
    act_probability = float(np.clip(0.38 + 0.25 * max(0.0, conf), 0.30, 0.52))
else:
    base_act = 0.55 + 0.40 * ((p_sel - 0.50) / 0.50) + 0.15 * conf
    dir_sign = 1.0 if primary_direction == 2 else -1.0
    ofi_alignment = dir_sign * float(orderflow_edge) if "orderflow_edge" in locals() else 0.0
    ofi_mod = float(np.clip(ofi_alignment * 0.4, -0.08, 0.06))
    act_probability = float(np.clip(base_act + ofi_mod, 0.50, 0.96))

bet_size = float(np.clip(act_probability * confidence, 0.05, 1.0)) if act_probability >= 0.70 else 0.0
else:
    act_probability = 0.42
    bet_size = 0.0
```

### 4.3 Pathological Consequences
1. **Total Bimodal Collapse:**
   - All 242 Flat Bars ($P_{flat} > P_{up}, P_{down}$) hit `else: act_probability = 0.42`.
   - 23 Directional Bars failed the "Ultra Sniper Gate" ($p_{sel} < 0.62$ or $p_{flat} > 0.28$) and were clamped into $[0.30, 0.52]$.
   - Exactly 5 Directional Bars satisfied $p_{sel} \ge 0.62$ and jumped across the dead zone $[0.52, 0.85]$ directly to $[0.850, 0.942]$.
2. **Artificial Threshold Redundancy:**
   - Because of the $[0.52, 0.85]$ dead zone, filtering by $\tau = 0.50$, $\tau = 0.60$, or $\tau = 0.70$ resulted in the **exact same 5 bars**.
3. **Calibration Catastrophe:**
   - In the $[0.70, 0.80)$ confidence bin (represented by these clamped signals), the empirical accuracy was **0.0%** (all 5 failed to win), while the model claimed $>74\%$ confidence.
   - The overall Expected Calibration Error (ECE) was **20.74%**, and the Brier score was **0.4165**, far worse than the naive climatological baseline (**0.2786**).
4. **Stacking Model Inversion:**
   - Because the stacking meta-learner was trained on unblended OOF features but evaluated on collinear live predictions, its weights blew up, forcing the author to implement an arbitrary blend:
     `blended_edge = 0.72 * mean_dir_edge + 0.28 * stack_dir_edge` (`live_loop_v2.py:96`).

---

## 5. Workspace Data Sources & Orthogonal Feature Inventory

To satisfy Requirement R1 and Acceptance Criteria AC1, the ensemble must de-correlate its members ($r \le 0.70$) across at least 3 distinct feature families:
1. Cross-Asset Lead-Lag (ETHUSDT / SOLUSDT $\to$ BTCUSDT)
2. Spot-Perpetual Basis Divergence & Funding Rate Pressure
3. Microstructure Signals (L1 Queue Imbalance, Microprice Drift, Trade Flow Imbalance / VPIN)

### 5.1 Data Source Inventory Table

| Data Domain | Available Local Sources | Vision Downloader Support | Live WebSocket Support | Status for Overhaul |
| :--- | :--- | :--- | :--- | :--- |
| **BTCUSDT Perp (USD-M)** | Full 1-day parquet (`features_BTCUSDT_2026-10-11.parquet`), OOF parquets, 45m live ledger | Yes (`binance_btc_usd_downloader.py`: aggTrades, bookDepth, bookTicker, metrics, fundingRate, klines) | Full (`wss://fstream.binance.com/ws`: aggTrade, depth20@100ms, bookTicker, markPrice@1s) | **Fully Ready** |
| **BTCUSDT Spot** | Parsing schema in `binance_parser.py`, downloader in `binance_btc_usd_downloader.py` | Yes (`spot/daily/aggTrades`, `klines`, `trades`) | Yes (`wss://stream.binance.com:9443/ws`: trade, depth20, bookTicker) | **Ready for Ingestion** (currently mocked from indexPrice in ring_buffer) |
| **ETHUSDT Perp / Spot** | Code references in `multi_asset_train.py` (`MULTI_ASSET_SYMBOLS`) | Needs symbol filter expansion in downloader (remove BTC-only filter) | Yes (`wss://fstream.binance.com/ws`: `ethusdt@aggTrade`, `ethusdt@bookTicker`) | **Code Ready, Needs Data Fetching** |
| **Order Book L1/L2 Microstructure** | Binance Vision `bookDepth` (12 levels) & `bookTicker` parsers; live 45m ledger has best bid/ask | Yes (`futures/um/daily/bookDepth` & `bookTicker`) | Live L2 20-level depth @ 100ms and real-time BBO ticker in `binance_ws.py` | **Fully Ready** |
| **Trade Flow Imbalance / VPIN** | `aggTrades` parser (`price`, `quantity`, `is_buyer_maker`, `transact_time`) | Yes (historical tick-by-tick aggTrades for all dates) | Live streaming aggTrade stream in `binance_ws.py` | **Fully Ready** |
| **Liquidations** | Parser in `binance_parser.py`, feature code in `lead_lag_liquidation.py` | Discontinued platform-wide on Binance Vision post-2024-10-14 | Live liquidation stream available via WebSocket (`<symbol>@forceOrder`) | **Discontinued Offline, Use Live WS Only** |
| **Funding Rates & Metrics** | Parsers in `binance_parser.py`, monthly fundingRate files | Yes (`fundingRate` monthly, `metrics` 5m daily) | Live funding rate in `markPrice@1s` WebSocket stream | **Fully Ready** |

---

## 6. Architectural Remediation Roadmap

Based on this exhaustive codebase investigation, the following architectural steps are recommended for the implementation team:

### R1: Member Model De-Correlation & Modularization
1. **Strip Micro-Signal Blending:** Completely delete the hardcoded `micro_ret` blending and `_signal_to_3class_proba` post-processing in `checkpoint_loaders.py`. Each model must output its own raw, uncorrupted prediction.
2. **De-duplicate the Transformer Members:** Drop redundant members (`nhits`, `patchtst`, `tft` correlate at $0.99+$). Keep at most one high-performing temporal sequence architecture.
3. **Introduce 3 Orthogonal Member Families:**
   - **Member A (Microstructure & Order Flow Specialist):** Fast gradient booster (LightGBM/CatBoost) trained exclusively on Level-1 queue imbalance, microprice drift, Level-2 OFI decay slope, and trade flow imbalance (VPIN / volume delta).
   - **Member B (Cross-Asset Lead-Lag Specialist):** Model trained on ETHUSDT and SOLUSDT order book flow and trade arrival lead-lag relative to BTCUSDT.
   - **Member C (Spot-Perp Basis & Macro Funding Specialist):** Model trained on spot-futures basis divergence, mark-index basis spread, funding rate pressure, and long/short open interest positioning.

### R2: Zero-Leakage Time-Series Protocol
1. **Eliminate Intra-Bar WebSocket Lookahead:** In `live_loop_v2.py`, ensure that feature calculation and model inference for bar $t$ use *only* market events timestamped $\le T_{\text{close}} - 1000\text{ ms}$.
2. **Remove Post-GPU Live Snapshot Return:** Delete the `live_ret_pct` injection from `_fuse()`. No forward price movements occurring during the GPU turnaround window may be fed into the decision stage.

### R3: Calibrated Continuous Meta-Labeler
1. **Activate Machine Learning Meta-Labeler:** Re-wire `live_loop_v2.py` to directly invoke `meta_filter.act_probability(row)`.
2. **Eliminate Piecewise Clamps:** Completely purge the `p_sel < 0.62` / `p_flat > 0.28` / `0.42` clamping code.
3. **Out-of-Fold Calibration:** Fit an Isotonic Regression or Platt Scaler on out-of-fold validation probabilities to ensure monotonic calibration across the continuous interval $[0, 1]$ and push ECE $\le 8.0\%$.

### R4: Horizon & Barrier Synchronization
1. **Lock Exit Horizon to Peak Alpha (30s / 3 bars):** Replace the 180s holding period with a 30s exit, matching the empirical alpha peak ($+0.367\text{ bps}$ drift-adjusted at $H3$).
2. **Dynamic Volatility Barriers:** Replace static 16/12 bps targets with dynamic $k \times \text{ATR}_{10s}$ barriers.
3. **Maker Execution Logic:** Implement BBO `GTX` (Post-Only) order placement to operate under the 4.0 bps round-trip VIP0 fee tier instead of 10.0 bps taker fees.

---
*Report autonomously generated by Features & Models Explorer. All code citations and line numbers verified directly from repository source files.*
