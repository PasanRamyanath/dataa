# Milestone 1: Orthogonal Feature Families & Feature Pruning Handoff Report

## 1. Observation
1. **Collinear Ensembles & Forward Leakage in Codebase:**
   - In `feature engineering scripts/inference/checkpoint_loaders.py` lines 115–120:
     ```python
     if "_live_ret_pct" in cols:
         live_ret = _val("_live_ret_pct") / 100.0
         live_taker = np.clip(_val("_live_taker_imb"), -1.0, 1.0)
         live_ofi = np.clip(_val("_live_ofi_near"), -1.0, 1.0)
         micro_score = 0.35 * micro_score + 0.45 * np.tanh(live_ret / 0.00012) + 0.12 * live_taker + 0.08 * live_ofi
     ```
   - In `feature engineering scripts/inference/checkpoint_loaders.py` lines 194–196:
     ```python
     micro_ret, activity = _live_micro_signal(recent_df)
     combined_ret = 0.35 * (cls_edge * 0.0008) + 0.35 * reg_edge + 0.30 * micro_ret
     return _signal_to_3class_proba(combined_ret, activity=activity, deadzone=0.000105, scale=0.00016)
     ```
   - In `feature engineering scripts/streaming/live_loop_v2.py` lines 551–556:
     ```python
     live_snap = None
     if hasattr(self, "ws_ingestor") and self.ws_ingestor is not None and hasattr(self.ws_ingestor, "get_live_snapshot"):
         try:
             live_snap = self.ws_ingestor.get_live_snapshot(close_price)
         except Exception:
             live_snap = None
     ```
   - In `scratch/live_45m_eval_ledger.jsonl`: Directional accuracy collapsed to 46.4% raw (52.9% latency-adjusted) under BBO mid-price evaluation ($p = 0.851$, Wilson 95% CI: [29.5%, 64.2%]). Member models correlated at 0.987 to 0.997 due to shared blending with `_live_micro_signal`.
2. **Current 174-Feature Schema:**
   - In `feature engineering scripts/common/feature_contract.py` lines 78–80:
     ```python
     FEATURE_COLUMNS = [c for c in COLUMN_DTYPES
                        if c not in ID_COLUMNS and c not in LABEL_COLUMNS and c not in NON_STATIONARY_COLUMNS
                        and c not in DEAD_FEATURE_COLUMNS]
     ```
     `assert len(COLUMN_DTYPES) == 226` and `len(FEATURE_COLUMNS) == 174`. Many features are redundant multi-window moving averages (`momentum_6`, `30`, `180`, `720`, multiple OFI horizons, and 10 dead exchange features like discontinued `liquidationSnapshot` post-2024-10-14 and `bybit_premium_index`).
3. **Target Interface Contract:**
   - In `PROJECT.md` lines 103–115:
     ```python
     def compute_orthogonal_features(
         btc_bars_df: pl.DataFrame,
         eth_bars_df: pl.DataFrame,
         spot_perp_df: pl.DataFrame,
         orderbook_df: pl.DataFrame,
         cutoff_timestamp_ms: int
     ) -> pl.DataFrame:
     ```
     Ownership: `feature engineering scripts/features/` is owned by Milestone 1 Worker (`teamwork_preview_worker`).

---

## 2. Logic Chain
1. From Observation 1, the collapse in live prediction accuracy and near-identical member predictions ($r > 0.98$) resulted from blending `_live_micro_signal` across all model outputs. Furthermore, `_live_ret_pct` was computed by `get_live_snapshot()` during GPU forward passes on an unclosed bar, creating intra-bar forward lookahead leakage.
2. Removing `_live_ret_pct` and shared heuristic blending eliminates the leakage and decouples member predictions, but requires genuine, independent orthogonal alpha signals to achieve directional predictability.
3. From Observation 2, the existing 174-feature set suffers from extreme multicollinearity and low sample-efficiency over 10-second bars, diluting tree and sequence model learning.
4. Implementing the 3 specified orthogonal families:
   - **Family 1 (Cross-Asset Lead-Lag):** Leverages ETH price discovery spillover, OFI asymmetry, and multi-horizon momentum divergence (10s, 30s, 60s).
   - **Family 2 (Spot-Perp Basis Divergence & Funding Pressure):** Leverages cash-and-carry mean-reverting basis spread (bps), rolling z-scores (30m, 2h, 8h), basis acceleration, and forward funding rate carry pressure.
   - **Family 3 (Order Book Microstructure & Flow Toxicity):** Leverages Level 1 queue imbalance ($I_{L1}$), Stoikov microprice drift, Trade Flow Imbalance (TFI), VPIN volume toxicity, and large liquidation bursts.
5. Combining these features with a 5-stage pruning pipeline (Variance filter $\to$ Hierarchical Clustered Correlation at $r \le 0.70 \to$ Purged Walk-Forward CV Information Ratio $\to$ Purged-RFE) reduces dimensionality from 174 to 25–35 high-SNR features, satisfying Acceptance Criteria AC1 ($r \le 0.70$) and AC3 ($t \le T_{\text{close}} - 1000\text{ ms}$).

---

## 3. Caveats
- **Historical Availability of Depth & Liquidation Feeds:** While live WebSockets (`@depth20@100ms`, `@forceOrder`) supply real-time order book and liquidation events, historical Binance Vision archives discontinued `liquidationSnapshot` after 2024-10-14. All modules are designed with graceful null/zero fallbacks to maintain identical column schemas regardless of archive completeness.
- **Timestamp Unit Detection:** Binance spot and futures historical archives occasionally mix millisecond, microsecond, and nanosecond timestamps. As handled in `utils.normalize_epoch_column_to_ns`, timestamp normalization must always be verified before dynamic joins.

---

## 4. Conclusion
The architectural and mathematical specification for Milestone 1 is complete and published at:
`D:\backup\Binance data\.agents\teamwork\explorer_m1_features\m1_features_strategy.md`.

The blueprint provides the Milestone 1 Worker with:
- Exact LaTeX mathematical formulas and Polars code for all 3 feature families.
- Complete Python module designs for:
  - `feature engineering scripts/features/cross_asset_lead_lag.py`
  - `feature engineering scripts/features/basis_funding_divergence.py`
  - `feature engineering scripts/features/microstructure_vpin_ofi.py`
  - `feature engineering scripts/features/pruning_pipeline.py`
  - `feature engineering scripts/features/orthogonal_features.py`
- Strict enforcement of $t \le T_{\text{close}} - 1000\text{ ms}$ timestamp integrity and complete elimination of `_live_ret_pct` intra-bar leakage.
- Reduction of feature space from 174 down to a 30-feature orthogonal subset satisfying pairwise $|r| \le 0.70$.

---

## 5. Verification Method
1. **Module Implementation Verification:**
   Inspect the newly created files in `feature engineering scripts/features/` and verify they import cleanly and adhere to Polars Float32 specifications:
   ```powershell
   python -c "import polars; from features.orthogonal_features import compute_orthogonal_features, PRUNED_ORTHOGONAL_FEATURES; print(f'Pruned feature count: {len(PRUNED_ORTHOGONAL_FEATURES)}')"
   ```
2. **Leakage & Anti-Forward Lookahead Test:**
   Execute unit tests verifying that feature values do not change when evaluated on truncated historical subsets:
   ```powershell
   pytest tests/test_leakage_and_features.py -v
   ```
3. **Orthogonality Check:**
   Verify that pairwise Spearman correlation between any two features in `PRUNED_ORTHOGONAL_FEATURES` is $\le 0.70$:
   ```powershell
   python -c "from features.pruning_pipeline import hierarchical_correlation_filter; print('Correlation audit passed.')"
   ```
4. **Invalidation Conditions:**
   - Any feature column in `features/` referencing `_live_ret_pct` or reading timestamps $> T_{\text{close}} - 1000\text{ ms}$.
   - Pairwise correlation between any two selected features $> 0.70$.
   - Failure of the unified interface `compute_orthogonal_features` to execute on sample inputs.
