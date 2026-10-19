# Handoff Report: Features & Models Exploration Survey
**Agent:** Features & Models Explorer  
**Working Directory:** `D:\backup\Binance data\.agents\teamwork\explorer_models_survey`  
**Handoff Type:** Hard (Survey Milestone Complete)  
**Primary Deliverable:** [`survey_models_report.md`](file:///D:/backup/Binance%20data/.agents/teamwork/explorer_models_survey/survey_models_report.md)

---

## 1. Observation

Direct code and telemetry observations from repository inspection:

### 1.1 Intra-Bar Lookahead Leakage
- **`inference/checkpoint_loaders.py:114–120`**:
  ```python
  if "_live_ret_pct" in cols:
      live_ret = _val("_live_ret_pct") / 100.0
      live_taker = np.clip(_val("_live_taker_imb"), -1.0, 1.0)
      live_ofi = np.clip(_val("_live_ofi_near"), -1.0, 1.0)
      micro_score = 0.35 * micro_score + 0.45 * np.tanh(live_ret / 0.00012) + 0.12 * live_taker + 0.08 * live_ofi
  ```
- **`streaming/live_ingestion/binance_ws.py:324–361`**:
  `get_live_snapshot(self, ref_price: float = 0.0)` computes `ret_pct = ((live_p - anchor) / anchor) * 100.0` where `anchor` is the close price of bar $t$ and `live_p` is sampled from the currently forming bar $t+1$.
- **`streaming/live_loop_v2.py:545–569`**:
  ```python
  predictions, latencies = self.coordinator.predict_tick(feature_df, timeout_seconds=8.5)
  live_snap = self.ws_ingestor.get_live_snapshot(close_price)
  fused = _fuse(predictions, ..., live_snap=live_snap, bar_dict=bar_dict)
  ```
- **`streaming/live_loop_v2.py:124–154`**:
  `live_ret_pct = float(snap.get("ret_pct", 0.0))` is combined into `net_signal = 0.55 * ens_dir_edge + 0.30 * orderflow_edge + 0.15 * np.tanh(live_ret_pct / 0.004)` and tilts Bayesian probabilities via `delta = float(np.clip(orderflow_edge * 0.06, -0.05, 0.05))` before determining `primary_direction`.

### 1.2 Member Model Prediction Collinearity ($r > 0.98$)
- **`inference/checkpoint_loaders.py:195, 277, 360, 489, 594`**:
  Every member predictor injects `_live_micro_signal(recent_df)`:
  - `_GbmFamilyPredictor`: `0.30 * micro_ret`
  - `_SentimentBranchPredictor`: `0.40 * micro_ret`
  - `_SsmChallengerPredictor`: `0.30 * micro_ret`
  - `_NeuralForecastPredictor` (`tft`, `patchtst`, `itransformer`, `nhits`): `0.45 * micro_ret`
  - `_KronosPredictor`: `0.50 * micro_ret`
- **`inference/checkpoint_loaders.py:415–491`**:
  `tft`, `patchtst`, `itransformer`, and `nhits` all share the identical wrapper class `_NeuralForecastPredictor`, computing trajectory returns with identical weighting (`0.50 * step1 + 0.30 * step3 + 0.20 * end`), identical centering (`0.70 * mean(history)`), and identical nonlinear mapping `_signal_to_3class_proba(combined_ret, activity=activity, deadzone=0.000105, scale=0.000165)`.
- **`fusion/oof/*.parquet` vs `result.md:230–238`**:
  In offline training parquets (where `_live_micro_signal` was *not* injected), raw member correlations are only $0.05$ to $0.52$. In the live ledger where `_live_micro_signal` was injected, correlations jumped to $0.987 \text{ to } 0.997$.

### 1.3 Meta-Labeler Bypass & Threshold Clamping
- **`fusion/meta_labeling.py:95–136`**:
  `MetaLabelFilter` defines `act_probability(df)` and `bet_size(df)` using a trained `LogisticRegression` model.
- **`streaming/live_loop_v2.py:186–201`**:
  The loaded `meta_filter` model is bypassed; instead, a hardcoded piecewise clamp is executed:
  ```python
  if p_sel < 0.62 or p_flat > 0.28:
      act_probability = float(np.clip(0.38 + 0.25 * max(0.0, conf), 0.30, 0.52))
  else:
      base_act = 0.55 + 0.40 * ((p_sel - 0.50) / 0.50) + 0.15 * conf
      ...
      act_probability = float(np.clip(base_act + ofi_mod, 0.50, 0.96))
      bet_size = float(np.clip(act_probability * confidence, 0.05, 1.0)) if act_probability >= 0.70 else 0.0
  else:
      act_probability = 0.42
      bet_size = 0.0
  ```
- **`scratch/live_45m_eval_ledger.jsonl`**:
  All 242 flat bars have `act_probability = 0.42`; 23 directional bars are in $[0.38, 0.47]$; exactly 5 bars are in $[0.85, 0.94]$. The interval $[0.50, 0.84]$ has zero samples.

### 1.4 Data Pipeline & Workspace Inventory
- **`common/feature_contract.py` & `col_schema.json`**:
  226 total columns = 1 `ts` + 9 labels + 32 non-stationary + 10 discontinued dead columns + 174 active stationary features.
- **`streaming/live_ingestion/ring_buffer.py:99–128`**:
  Synthetic mocks in live ring buffer: `spot_df` uses `index_price`, `bybit_df` uses Binance volumes, `implied_vol` uses `realized_vol_360 * ann_factor * 0.5 + 45.0`.
- **`binance_btc_usd_downloader.py`**:
  Currently restricted to BTC instruments (`is_btc_usd_symbol()`). ETHUSDT data code exists in `resource_maximization/multi_asset_train.py` but data has not been fetched locally.

---

## 2. Logic Chain

1. **Intra-Bar Lookahead Leakage:**
   - In `binance_ws.py`, `get_live_snapshot(close_price)` computes the percentage return between the completed bar close and trades arriving during the ~1000 ms GPU turnaround window.
   - Because $H1$ prediction is defined as $(P_{t+1} - P_t) / P_t$, the live snapshot samples $10\%$ into the future return period.
   - Injecting this return into `_live_micro_signal` and `_fuse()` gave the sentiment branch and live decisions an artificial, non-causal lookahead bias. When evaluated with strict latency adjustment ($t + 1018\text{ ms}$), the edge drops from $+0.227\text{ bps}$ to $-0.176\text{ bps}$ ($p = 1.000$).

2. **Member Collinearity Collapse:**
   - Offline training generated diverse models with raw pairwise correlations between $0.05$ and $0.52$.
   - In live deployment, `checkpoint_loaders.py` injected 30% to 50% `_live_micro_signal` into every member's output.
   - Furthermore, `tft`, `patchtst`, `itransformer`, and `nhits` shared identical trajectory extraction, centering, and sigmoid mapping.
   - This shared wrapper layer overwhelmed individual model variance, causing predictions to correlate at $> 0.98$.

3. **Meta-Labeler Bimodal Dead Zone:**
   - The stacking meta-learner, fed collinear inputs at runtime, produced pathological coefficients (inverted weights and negative intercept on flat).
   - Rather than fixing the collinearity, the author implemented the "Ultra Sniper Gate" heuristic in `live_loop_v2.py`.
   - This forced all flat bars to $0.420$ and non-conforming bars to $[0.30, 0.52]$, while conforming bars jumped directly to $[0.85, 0.94]$.
   - This destroyed probability calibration (ECE exploded to $20.74\%$, Brier score collapsed to $0.4165$ vs climatological baseline $0.2786$) and made thresholds $\tau \in [0.5, 0.7]$ identical filters.

4. **Data Readiness for Orthogonal Signals:**
   - The infrastructure for Binance USD-M futures (L1 bookTicker, L2 bookDepth 20 levels @ 100ms, tick aggTrades) is already built and working in `binance_ws.py`.
   - Cross-asset signals (ETHUSDT / SOLUSDT) can be added immediately via WebSocket streams (`ethusdt@aggTrade`, `ethusdt@bookTicker`) and by expanding `binance_btc_usd_downloader.py`.

---

## 3. Caveats

1. **Historical Depth Sample Staleness:** In historical Binance Vision archives, `bookDepth` snapshots occur at $\approx 30\text{s}$ cadence rather than the $100\text{ms}$ available in the live WebSocket stream. Model training on historical depth must account for this difference in granularity.
2. **Kaggle GPU Training Constraints:** The deep sequence models (TFT, PatchTST, SSM, Kronos) require GPU acceleration (T4/V100/A100) and cannot be trained on local CPU in reasonable time. A lightweight gradient boosting approach (LightGBM/CatBoost) should be prioritized for rapid orthogonal feature validation.
3. **External Exchange Feeds:** Bybit data in `bybit_btc_usd_downloader.py` relies on Bybit's historical API, which occasionally encounters rate limiting. Cross-asset Binance streams (e.g. ETHUSDT and SOLUSDT on Binance USD-M) offer significantly higher reliability and co-location latency advantages.

---

## 4. Conclusion

The pipeline's failure was caused by architectural coupling and uncalibrated heuristic overrides, rather than fundamental alpha absence:
- Intra-bar leakage in `get_live_snapshot` and `_live_micro_signal` gave false confidence during prototyping.
- Wrapper-level injection of `_live_micro_signal` destroyed ensemble diversity, collapsing 8 member models into a single redundant vote ($r > 0.98$).
- Bypassing the ML meta-labeler with piecewise threshold clamping created an uncalibrated bimodal distribution ($ECE = 20.74\%$).

Remediating these issues requires:
1. Stripping `_live_micro_signal` out of member model wrappers and deleting the intra-bar `live_ret_pct` leak.
2. Pruning the redundant transformer models and introducing 3 genuinely orthogonal signal families: Cross-Asset Lead-Lag (ETH/SOL), Spot-Perp Basis/Funding, and Microstructure Queue/VPIN.
3. Restoring continuous, isotonic-calibrated meta-labeling without piecewise clamps.
4. Synchronizing holding horizon (30s) and dynamic volatility barriers with maker execution.

---

## 5. Verification Method

To independently verify these findings, execute the following commands in powershell:

```powershell
# 1. Reproduce the quantitative audit metrics, correlations, ECE, and p-values
python "feature engineering scripts/scratch/rigorous_evaluation_audit.py"

# 2. Inspect the intra-bar return leak in live ingestion:
# File: feature engineering scripts/streaming/live_ingestion/binance_ws.py (lines 344-356)
# File: feature engineering scripts/streaming/live_loop_v2.py (lines 142-154, 552-556)
# File: feature engineering scripts/inference/checkpoint_loaders.py (lines 114-120)

# 3. Inspect member model wrapper micro-signal injection:
# File: feature engineering scripts/inference/checkpoint_loaders.py (lines 195, 277, 360, 489, 594)

# 4. Inspect the meta-labeler piecewise clamp:
# File: feature engineering scripts/streaming/live_loop_v2.py (lines 189-201)

# 5. Verify raw OOF correlation vs live collinearity:
python -c "
import sys; sys.path.insert(0, 'feature engineering scripts')
from common.oof import load_all_oof, load_regime_states
from fusion.stacking_meta_learner import build_meta_features
import numpy as np

meta_df = build_meta_features(load_all_oof(), load_regime_states())
m1 = (meta_df['tft__prob_up'] - meta_df['tft__prob_down']).to_numpy()
m2 = (meta_df['nhits__prob_up'] - meta_df['nhits__prob_down']).to_numpy()
print('Raw OOF Correlation (TFT vs NHITS):', np.corrcoef(m1, m2)[0, 1])
"
# Result: ~0.1934 (raw OOF) vs 0.997 (live ledger after wrapper blending)
```

**Invalidation Conditions:**
- If removing `_live_micro_signal` from `checkpoint_loaders.py` causes live member correlations to remain $> 0.90$, collinearity is driven by input features rather than the post-processing wrapper.
- If `get_live_snapshot` can be proven to only read trades timestamped $\le T_{\text{close}} - 1000\text{ ms}$, the intra-bar lookahead claim would be invalidated. (Direct code inspection proves it samples during the GPU forward pass post-close).
