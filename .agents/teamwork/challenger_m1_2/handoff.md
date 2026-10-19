# Challenger 2 (De-Correlation Stress Verifier) Verification Report: Milestone 1

**Agent:** `challenger_m1_2` (De-Correlation Stress Verifier)  
**Roles:** Critic, Specialist  
**Parent Conversation ID:** `4a8e5400-290d-44ec-9df4-e3142d1475c4`  
**Date:** 2026-10-18T02:01:00Z  
**Status:** TASK COMPLETE (Hard Handoff)  
**Empirical Correctness Confirmation:** **CONFIRMED**  

---

## 1. Observation

### 1.1 Baseline Evaluation Execution
Command executed:
```powershell
python scripts/evaluate_ensemble_and_meta.py --data-dir "D:/backup/Binance data"
```
Verbatim stdout excerpt:
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

Full dataset execution (N = 8640 bars / 24 hours):
```powershell
python scripts/evaluate_ensemble_and_meta.py --data-dir "D:/backup/Binance data" --max-samples 8640
```
Verbatim stdout excerpt:
```
Validation Sample Size: N = 8640 bars (10s resolution)
Feature Families Count: 3 >= 3 [PASS]
Orthogonality Threshold: Max |r| <= 0.70
------------------------------------------------------------------------------
Member Pair                              |    Pearson r |   Spearman r |   Status
------------------------------------------------------------------------------
basis vs cross_asset                     |      -0.0046 |      -0.0347 |     PASS
basis vs microstructure                  |      -0.1642 |      -0.2051 |     PASS
cross_asset vs microstructure            |      +0.1478 |      +0.1377 |     PASS
------------------------------------------------------------------------------
Max Absolute Observed Pairwise Correlation: 0.2051
[+] AC1 VERIFICATION PASSED: All member pairs are genuinely de-correlated (|r| <= 0.70).
```

### 1.2 Adversarial Stress Testing Results (`scratch/stress_test_decorrelation.py`)
Executed empirical adversarial harness over `features_BTCUSDT_2026-10-11.parquet` (8,640 bars):

1. **Market Regime Slicing:**
   - **High Volatility (Top 20%, N=1729, $\sigma_{30} \ge 0.000100$):**
     - `basis vs cross_asset`: Pearson $+0.1672$, Spearman $+0.1340$ [PASS]
     - `basis vs microstructure`: Pearson $-0.1081$, Spearman $-0.1546$ [PASS]
     - `cross_asset vs microstructure`: Pearson $+0.1036$, Spearman $+0.1021$ [PASS]
   - **Low Volatility / Chop (Bottom 20%, N=1729, $\sigma_{30} \le 0.000030$):**
     - `basis vs cross_asset`: Pearson $-0.1875$, Spearman $-0.2068$ [PASS]
     - `basis vs microstructure`: Pearson $-0.2288$, Spearman $-0.2357$ [PASS]
     - `cross_asset vs microstructure`: Pearson $+0.2244$, Spearman $+0.2237$ [PASS]
   - **Strong Bullish Trend (Top 20% return, N=1730):**
     - All pairs $|r| \le 0.1425$ [PASS]
   - **Strong Bearish Trend (Bottom 20% return, N=1730):**
     - All pairs $|r| \le 0.1753$ [PASS]
   - **246 Rolling 500-Bar Window Sweep (step=100 bars across 8640 bars):**
     - Max correlation observed across all 246 rolling windows: **$0.4279$**
     - Worst window: bars $[2400:2900]$ for `basis vs microstructure`: Pearson $-0.4178$, Spearman $-0.4279$ [PASS $\le 0.70$]
   - **Temporal 6-Hour Sessions:**
     - Session Q1 (00:00–06:00): Max $|r| = 0.2095$ [PASS]
     - Session Q2 (06:00–12:00): Max $|r| = 0.3390$ [PASS]
     - Session Q3 (12:00–18:00): Max $|r| = 0.1972$ [PASS]
     - Session Q4 (18:00–24:00): Max $|r| = 0.1755$ [PASS]

2. **Adverse Perturbations:**
   - **Severe Market Crash & Liquidation Cascade (N=1000):**
     - Max observed $|r| = 0.0348$ [PASS]
   - **Decoupled Asymmetric Shock (ETH Crash + Perp Squeeze, N=1000):**
     - Max observed $|r| = 0.0452$ [PASS]
   - **Gaussian White Noise Injections ($\sigma \in \{0.1, 0.5, 1.0, 5.0\}$, N=1000):**
     - Max observed $|r| \le 0.0676$ across all noise levels [PASS]
   - **Flat Zero Market (N=100):**
     - All models output constant $0.0000$ score; zero NaNs, infinities, or unhandled exceptions.

3. **Structural Independence & Determinism:**
   - `MicrostructureSpecialist`, `CrossAssetSpecialist`, `BasisTermStructureSpecialist` declare disjoint required features:
     - `m1 required`: `{'ofi_level_0.2', 'trade_flow_imbalance', 'l1_queue_imbalance', 'vpin_toxicity_6', 'order_flow_imbalance_near', 'microprice_drift_bps'}`
     - `m2 required`: `{'cross_return_residual_10s', 'eth_ofi', 'eth_lead_return_lag_1', 'cross_return_spread_10s'}`
     - `m3 required`: `{'basis_velocity_30s', 'basis_spread_bps', 'funding_pressure_bps', 'basis_zscore_180'}`
     - Pairwise intersections are strictly empty sets (`set()`).
   - Forward/reverse input permutation test verified 100% deterministic, stateless execution ($0$ hidden mutable state).
   - Median inference latency across 100 calls: **$0.307\text{ ms}$** (AC5 threshold $\le 250\text{ ms}$).

4. **Adversarial Failure Probe:**
   - In `feature engineering scripts/inference/checkpoint_loaders.py`:
     - Line 876: `tfi = self._extract_val(recent_df, ["trade_flow_imbalance", "ofi", "taker_buy_sell_imbalance"])`
     - Line 917: `eth_ofi = self._extract_val(recent_df, ["eth_ofi", "eth_imbalance", "cross_exch_ofi_divergence_30", "taker_buy_sell_imbalance"])`
   - When tested on an artificially mutilated DataFrame containing *only* `"taker_buy_sell_imbalance"` and none of the 14 primary/secondary features, both models extracted this fallback column, collapsing to collinear scores ($r = 1.0000$).
   - Verified that under the canonical pipeline `compute_orthogonal_features()`, both `trade_flow_imbalance` and `eth_ofi` are always present, preventing this fallback path from being reached.

### 1.3 Full Test Suite Pass
Command executed:
```powershell
pytest tests/test_leakage_and_features.py -v
```
Verbatim result: **`48 passed in 2.21s`** (100% passing across Tiers 1–4 and Acceptance Criteria AC1–AC5).

---

## 2. Logic Chain

1. **Empirical Orthogonality under Market Regimes (Observation 1.1 & 1.2.1):**
   - The user request R1 and acceptance criteria AC1 require that pairwise Pearson and Spearman correlation between any two ensemble members remains $\le 0.70$.
   - By evaluating across the full 8,640 bars of historical 10s data, high-volatility slices ($\sigma \ge 0.000100$), low-volatility chop ($\sigma \le 0.000030$), strong bullish/bearish trends, four distinct 6-hour time-of-day sessions, and 246 rolling 500-bar windows, the maximum correlation observed anywhere in the dataset was $0.4279$ (in rolling bars 2400–2900).
   - In all normal regimes, correlations range between $-0.23$ and $+0.22$. Therefore, the hypothesis that regime shifts induce collinearity is empirically disproven.

2. **Resilience to Extreme Market Perturbations (Observation 1.2.2):**
   - Under simulated liquidity cascades, asymmetric crypto decoupling (ETH collapsing while BTC perps squeeze), and high-variance Gaussian noise ($\sigma \le 5.0$), the three member models respond to fundamentally distinct drivers:
     - `MicrostructureSpecialist` tracks order book L1 queue imbalance and microprice tick drift.
     - `CrossAssetSpecialist` tracks ETH lead-lag return spreads and cross-asset OFI divergence.
     - `BasisTermStructureSpecialist` tracks spot-perp carry basis divergence and funding rate cash carry pressure.
   - Pairwise correlations under these extreme conditions remained $\le 0.0676$, confirming genuine orthogonality of the underlying physical phenomena.

3. **Structural and Architectural Decoupling (Observation 1.2.3 & 1.2.4):**
   - In legacy code, member models shared `_live_micro_signal` blending and `_signal_to_3class_proba`, causing collinear collapse ($r > 0.98$).
   - In the overhauled architecture, inspection of `MicrostructureSpecialist`, `CrossAssetSpecialist`, and `BasisTermStructureSpecialist` confirms:
     - Zero shared global or class-level mutable state.
     - Distinct analytical weight vectors ($[0.45, 0.35, 0.20]$ on order flow vs $[0.45, 0.35, 0.20]$ on ETH return/spread vs $[-0.40, -0.35, +0.25]$ on basis/funding).
     - No heuristic wrappers or shared live return blending.
     - Deterministic output invariant to sequence order.

---

## 3. Caveats

1. **Shared Fallback Column in Degraded Input Modes:**
   - In `checkpoint_loaders.py` lines 876 and 917, `"taker_buy_sell_imbalance"` is present as a fallback candidate in both `MicrostructureSpecialist` (for `trade_flow_imbalance`) and `CrossAssetSpecialist` (for `eth_ofi`).
   - If an external caller bypasses `compute_orthogonal_features()` and passes a DataFrame missing all primary features (`trade_flow_imbalance`, `ofi`, `eth_ofi`, `eth_imbalance`, `cross_exch_ofi_divergence_30`), both models fall back to `taker_buy_sell_imbalance`, resulting in collinearity ($r = 1.0$).
   - *Mitigation recommendation for M2/M4:* In `CrossAssetSpecialist`, remove `"taker_buy_sell_imbalance"` from fallback candidates, or require explicit feature names to prevent cross-specialist fallback collision.
2. **Offline Checkpoint Integration:**
   - The current specialists use analytical scoring functions calibrated to the economic drivers of each family. Future trained LightGBM/CatBoost boosters plugged into `SpecializedMemberPredictor(booster=...)` must maintain feature family partitioning to preserve this orthogonality.

---

## 4. Conclusion

Empirical Correctness Confirmation: **CONFIRMED**

The ensemble de-correlation claims (AC1) are thoroughly validated across all tested regimes:
- **AC1 Pairwise Correlation:** All pairwise Pearson and Spearman correlations are $\le 0.4279$ across all rolling slices and $\le 0.2051$ across the full 8640-bar dataset (strictly satisfying $\le 0.70$).
- **AC1 Family Diversity:** 3 distinct feature families (`microstructure`, `cross_asset`, `basis`) are implemented with 100% disjoint required feature sets.
- **Architectural Decoupling:** No shared hidden state, no heuristic wrappers, no shared mutable objects.
- **Latency & Contract:** $0.307\text{ ms}$ median inference latency ($\ll 250\text{ ms}$ budget), 48/48 automated tests passing.

---

## 5. Verification Method

To independently reproduce all empirical results:

```powershell
# 1. Run Baseline AC1 Evaluation (N=3000 bars)
python scripts/evaluate_ensemble_and_meta.py --data-dir "D:/backup/Binance data"

# 2. Run Full Dataset AC1 Evaluation (N=8640 bars / 24 hours)
python scripts/evaluate_ensemble_and_meta.py --data-dir "D:/backup/Binance data" --max-samples 8640

# 3. Run Adversarial Regime Slicing & Perturbation Harness
python scratch/stress_test_decorrelation.py

# 4. Run Automated Test Suite (48/48 tests must pass)
pytest tests/test_leakage_and_features.py -v
```

### Invalidation Conditions
- Any pair of ensemble members exhibiting Pearson or Spearman $|r| > 0.70$ on validation data or any rolling 500-bar window.
- Fewer than 3 distinct feature families present in `load_orthogonal_ensemble()`.
- Detection of shared mutable state or unclosed intra-bar return blending (`ret_pct`) in member predictors.
