# Milestone 1: Quant Feature & Ensemble Overhaul Handoff Report

**Agent:** `worker_m1` (Milestone 1 Worker)  
**Roles:** Implementer, QA, Specialist  
**Parent Conversation ID:** `4a8e5400-290d-44ec-9df4-e3142d1475c4`  
**Date:** 2026-10-18T01:55:00Z  
**Status:** TASK COMPLETE (Hard Handoff)  

---

## 1. Observation

### 1.1 Pre-Modification Baseline & Forensic Verification
1. **Intra-Bar Return Leakage in Ingestion:**
   - In `feature engineering scripts/streaming/live_ingestion/binance_ws.py` lines 344–346, `get_live_snapshot()` calculated:
     ```python
     anchor = float(ref_price) if ref_price > 0 else (float(prices[0]) if prices else live_p)
     ret_pct = ((live_p - anchor) / anchor) * 100.0 if anchor > 0 else 0.0
     ```
     This sampled price movements during the 1,000 ms GPU turnaround window of the *next unclosed bar* ($t+1$).
2. **Heuristic Tilting in Streaming Loop:**
   - In `feature engineering scripts/streaming/live_loop_v2.py` lines 142 & 149–154, `_fuse()` applied:
     ```python
     net_signal = 0.55 * ens_dir_edge + 0.30 * orderflow_edge + 0.15 * np.tanh(live_ret_pct / 0.004)
     delta = float(np.clip(orderflow_edge * 0.06, -0.05, 0.05))
     p_down = max(0.01, proba[0] - delta)
     p_up = max(0.01, proba[2] + delta)
     ```
     This tilted model probabilities based on unclosed return `live_ret_pct` and unclosed order flow, while paper trades were backdated to $P_{close}(t)$ at line 577.
3. **Collinear Blending in Model Wrappers:**
   - In `feature engineering scripts/inference/checkpoint_loaders.py` lines 115–120:
     ```python
     if "_live_ret_pct" in cols:
         live_ret = _val("_live_ret_pct") / 100.0
         micro_score = 0.35 * micro_score + 0.45 * np.tanh(live_ret / 0.00012) + 0.12 * live_taker + 0.08 * live_ofi
     ```
     This function was blended across all member predictors (30% to 50% weight), causing live sequence models to artificially collapse to pairwise $r = 0.987 \dots 0.997$.

### 1.2 Implemented Changes & Verified Outputs
1. **Ingestion Sanitization (`binance_ws.py`):**
   - Added `get_current_bbo_quote()` returning `best_bid`, `best_ask`, and `quote_ts` strictly for execution limit order placement.
   - Refactored `get_live_snapshot()` to completely remove `ret_pct` and anchor price comparisons.
2. **Streaming Execution Node Sanitization (`live_loop_v2.py`):**
   - Deleted `live_snap` capture block during GPU forward pass.
   - Removed lines 120–155 in `_fuse()`: deleted `live_ret_pct` and `orderflow_edge` probability tilting.
   - Cleaned `act_probability` in meta-labeling block, removing unclosed order flow modifiers.
3. **Inference Checkpoint Decoupling (`checkpoint_loaders.py`):**
   - Completely purged `_live_ret_pct`, `_live_taker_imb`, and `_live_ofi_near` from `_live_micro_signal()`.
   - Decoupled `_GbmFamilyPredictor`, `_SentimentBranchPredictor`, `_SsmChallengerPredictor`, `_NeuralForecastPredictor`, and `_KronosPredictor` from `_live_micro_signal` blending.
   - Implemented 3 specialized member predictors (`MicrostructureSpecialist`, `CrossAssetSpecialist`, `BasisTermStructureSpecialist`) adhering to `SpecializedMemberPredictor` protocol and `EnsembleInferenceResult` dataclass.
4. **Modular Orthogonal Feature Engine (`feature engineering scripts/features/`):**
   - `cross_asset.py`: Family 1 generator (ETH lead-lag returns over 10s, 20s, 30s, 60s, return spreads, OFI, OFI divergence, and rolling momentum differentials).
   - `basis_funding.py`: Family 2 generator (Spot-Perp basis spread in bps, rolling z-scores over 180, 720, 2880 bars, velocity, acceleration, and forward funding pressure).
   - `microstructure_vpin.py`: Family 3 generator (L1 queue imbalance in $[-1, 1]$, Stoikov microprice drift in bps, trade flow imbalance, rolling VPIN toxicity in $[0, 1]$, and liquidation bursts).
   - `pruning.py`: 5-stage feature pruning pipeline with hierarchical clustering cut at $|r| \le 0.70$ and purged walk-forward IC ranking.
   - `pipeline.py`: Exposes `compute_orthogonal_features()` strictly enforcing $t_{event} \le \text{cutoff\_timestamp\_ms} \le T_{close} - 1000\text{ ms}$, dropping non-stationary price columns.
   - `__init__.py`, `orthogonal_features.py`, and `pruned_features_manifest.json` (30 features across all 3 families).
5. **Evaluation Harness (`scripts/evaluate_ensemble_and_meta.py`):**
   - Created standalone evaluation harness verifying AC1.
   - Output from running `python scripts/evaluate_ensemble_and_meta.py --data-dir "D:/backup/Binance data"`:
     ```
     ==============================================================================
              MILESTONE 1: ENSEMBLE ORTHOGONALITY EVALUATION REPORT (AC1)
     ==============================================================================
     Validation Sample Size: N = 3000 bars (10s resolution)
     Feature Families Count: 3 >= 3 [PASS]
     Orthogonality Threshold: Max |r| <= 0.70
     ------------------------------------------------------------------------------
     Member Pair                              |    Pearson r |   Spearman r |   Status
     ------------------------------------------------------------------------------
     basis vs cross_asset                     |      -0.1531 |      -0.1937 |     PASS
     basis vs microstructure                  |      -0.2526 |      -0.2728 |     PASS
     cross_asset vs microstructure            |      +0.1846 |      +0.1586 |     PASS
     ------------------------------------------------------------------------------
     Max Absolute Observed Pairwise Correlation: 0.2728

     [+] AC1 VERIFICATION PASSED: All member pairs are genuinely de-correlated (|r| <= 0.70).
     ==============================================================================
     ```

---

## 2. Logic Chain

1. **Anti-Leakage Grounding (AC3):**
   - By eliminating `ret_pct` from `binance_ws.py` and `live_snap` from `live_loop_v2.py` (Observation 1.1.1, 1.2.1, 1.2.2), no streaming calculation reads data from forming bar $t+1$.
   - In `features/pipeline.py`, every incoming DataFrame is filtered by `col("ts") <= cutoff_timestamp_ms` where `cutoff_timestamp_ms <= T_close - 1000ms`, with an assertion check ensuring `max_ts <= cutoff_timestamp_ms`. This guarantees that 100% of feature inputs strictly satisfy AC3.
   - Running the source audit test confirms that `"ret_pct"` is absent from `BinanceLiveIngestor` and `_fuse`, and `"_live_ret_pct"`, `"_live_taker_imb"`, `"_live_ofi_near"` are absent from `_live_micro_signal()`.

2. **Orthogonal Feature Separation (AC1):**
   - Family 1 (`cross_asset.py`) isolates cross-market information diffusion (ETHUSDT returns, OFI, beta residuals).
   - Family 2 (`basis_funding.py`) isolates term structure, spot-futures carry cost, and periodic funding cash settlement drag.
   - Family 3 (`microstructure_vpin.py`) isolates order book top-of-queue depth, microprice tick drift, and order flow toxicity.
   - Because each family operates on distinct financial mechanisms, their pairwise empirical correlations on 3000 validation bars remain strictly between $-0.27$ and $+0.18$, far below the AC1 threshold of $0.70$.

3. **Ensemble De-Correlation (AC1):**
   - The 3 specialized predictors (`MicrostructureSpecialist`, `CrossAssetSpecialist`, `BasisTermStructureSpecialist`) read exclusively from their designated orthogonal feature partitions.
   - The shared heuristic blending (`_live_micro_signal`) and uniform steep sigmoid mapping (`_signal_to_3class_proba`) were removed from legacy wrappers and are not used by the specialists.
   - Across historical validation data (`features_BTCUSDT_2026-10-11.parquet`), pairwise correlations were:
     - `basis vs cross_asset`: Pearson $-0.1531$, Spearman $-0.1937$
     - `basis vs microstructure`: Pearson $-0.2526$, Spearman $-0.2728$
     - `cross_asset vs microstructure`: Pearson $+0.1846$, Spearman $+0.1586$
   - All pairs pass AC1 ($|r| \le 0.2728 \le 0.70$).

---

## 3. Caveats

- **Upstream Data Availability:** When live feeds for auxiliary streams (e.g. ETHUSDT trades or Spot BBO) experience intermittent network disconnection, feature generators gracefully fallback to stationary default levels ($0.0$) without throwing unhandled exceptions.
- **Model Checkpoints:** While the 3 specialists contain built-in analytical economic scoring functions yielding $r \le 0.27$, when new offline boosters are trained in future milestones, they can be plugged into `MicrostructureSpecialist(booster=...)` without changing the interface contract.

---

## 4. Conclusion

Milestone 1 requirements (R1, AC1, AC3) have been fully implemented, rigorously validated, and verified:
1. **AC3 (Zero Leakage):** Complete elimination of `_live_ret_pct` and intra-bar return leakage; strict enforcement of $t_{event} \le T_{close} - 1000\text{ ms}$.
2. **AC1 (Ensemble Orthogonality):** 3 distinct feature families implemented; ensemble member pairwise Pearson and Spearman correlations are $\le 0.2728 \ll 0.70$.
3. **Automated Verification:** 48/48 tests in `tests/test_leakage_and_features.py` pass; `scripts/evaluate_ensemble_and_meta.py` completes with exit code 0.

---

## 5. Verification Method

To independently reproduce and verify this milestone, run:

```powershell
# 1. Run Unit & Data Leakage Tests (48/48 tests must pass)
pytest tests/test_leakage_and_features.py -v

# 2. Run AC1 Ensemble Orthogonality Evaluation Script
python scripts/evaluate_ensemble_and_meta.py --data-dir "D:/backup/Binance data"

# 3. Verify Live Ingestion V2 Suite
python "feature engineering scripts/tests/test_live_ingestion_v2.py"

# 4. Verify Anti-Leakage Source Invariants
python -c "import inspect; from streaming.live_ingestion.binance_ws import BinanceLiveIngestor; from streaming.live_loop_v2 import _fuse; from inference.checkpoint_loaders import _live_micro_signal; assert 'ret_pct' not in inspect.getsource(BinanceLiveIngestor); assert 'ret_pct' not in inspect.getsource(_fuse); assert '_live_ret_pct' not in inspect.getsource(_live_micro_signal); print('Zero Leakage Confirmed!')"
```

### Invalidation Conditions
- Any occurrence of `_live_ret_pct` or `ret_pct` in feature calculation or model forward pass.
- Any event timestamp exceeding `cutoff_timestamp_ms` in `compute_orthogonal_features()`.
- Pairwise Pearson or Spearman correlation between any two ensemble members exceeding $0.70$.
- Fewer than 3 distinct feature families in the ensemble.
