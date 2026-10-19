# Original User Request

## 2026-10-18T01:18:32Z

# Teamwork Project Prompt — BTCUSDT 10-Second Alpha & Execution Overhaul

> Status: Ready for launch — approved by user
> Goal: Multi-agent overhaul of BTCUSDT 10s prediction & maker execution pipeline
> Target Team: Architecture, Quant ML Engineer, Execution Engine Specialist, Adversarial Auditor
> Integrity Mode: Development (Strict verification & validation)

## Project Overview & Objective
Complete architectural and algorithmic overhaul of the BTCUSDT 10-second-bar futures prediction and execution pipeline in `D:\backup\Binance data`.

Empirical audit of the 45-minute live prediction ledger (`scratch/live_45m_eval_ledger.jsonl`) established that previously claimed 60%–64% directional accuracy collapsed to 46.4% raw (59.1% non-tie, 52.9% latency-adjusted) when evaluated against true BBO mid-price, which is statistically indistinguishable from noise ($p = 0.851$, Wilson 95% CI: $[29.5\%, 64.2\%]$). Furthermore:
1. Member models (NHITS, PatchTST, TFT, SSM) correlate at 0.987 to 0.997, acting as a single redundant model.
2. The meta-labeler exhibited bimodal collapse due to artificial threshold clamping (`p_sel < 0.62` clamped to 0.42, 5 forced high-confidence bars).
3. Alpha peaks at 20–30s (3 bars) with mean move $\approx 0.74\text{ bps}$, whereas execution held trades for 180s against unreachable 16/12 bps barriers, causing 100% time-stop expirations and net losses against secular drift.
4. Intra-bar lookahead leakage existed in the sentiment branch (`_live_micro_signal` reading unclosed bar returns).

The objective is to deliver a production-grade, leak-free system with genuine orthogonal alpha, calibrated continuous meta-labeling, volatility-scaled barriers, and a BBO queue-aware maker execution node validated under purged walk-forward cross-validation.

Working directory: `D:\backup\Binance data`

---

## Requirements

### R1. Orthogonal Multi-Architecture Ensemble & Feature De-Duplication
- **De-correlate Member Predictions**: Replace collinear transformer models (NHITS, PatchTST, TFT, SSM currently $r > 0.98$) with genuinely diverse signal generators:
  - Lead-lag cross-asset dynamics (e.g., ETHUSDT lead-lag vs BTCUSDT).
  - Spot-perpetual basis divergence and funding rate pressure.
  - Microstructure signals: Level 1 queue imbalance, microprice drift, trade flow imbalance (VPIN / order flow toxicity), and large liquidation bursts.
- **Strict Time-Series Integrity & Leakage Prevention**: Ensure every feature input is strictly known prior to bar close ($t \le T_{\text{close}}$) with timestamp-verified latency offsets ($\ge 1000\text{ ms}$). Eliminate any intra-bar lookahead leakage (such as unclosed bar returns in the sentiment/micro-signal branch).
- **Dimensionality & Regularization**: Prune the current 174-feature set down to high signal-to-noise orthogonal features using mutual information or recursive feature elimination with purged walk-forward cross-validation to prevent overfitting.

### R2. Continuous Calibrated Meta-Labeler & Volatility Hurdle Engine
- **Continuous Probabilistic Meta-Labeling**: Rebuild the meta-labeling secondary model to output unconstrained, smoothly calibrated confidence probabilities $\hat{p} \in [0, 1]$. Completely remove artificial piecewise threshold clamps (e.g., constant 0.42 branches).
- **Probability Calibration**: Calibrate model probabilities using isotonic regression or Platt scaling on out-of-fold validation data to achieve Expected Calibration Error (ECE) $\le 8\%$ and improve upon the climatological Brier score (0.2786).
- **Cost & Magnitude Hurdle**: Enforce that trades are only authorized if the predicted absolute price move $|\Delta \hat{p}|$ comfortably exceeds round-trip transaction costs plus an edge buffer ($|\Delta \hat{p}| \ge \text{Fee}_{\text{round-trip}} + \text{Buffer}$).
- **Realized Volatility Gate**: Integrate an Average True Range (ATR) or realized volatility gate that halts directional authorization during low-volatility compression regimes (e.g. 10s mean absolute move $< 1.0\text{ bps}$).

### R3. BBO Queue-Aware Maker Execution Node & Barrier Synchronization
- **Horizon & Barrier Synchronization**: Align position holding horizon and profit-taking barriers with the empirical alpha decay curve (holding horizon = 3 bars / 30 seconds). Replace static 16/12 bps barriers with dynamic volatility-scaled barriers (e.g. $k \times \text{ATR}_{10s}$).
- **Pure Maker-Post Execution**: Formulate execution as limit order placement at the BBO (Best Bid / Best Offer) using `GTX` (Post-Only) order types to earn maker fee tier rates (e.g., Binance USD-M VIP0 maker fee of 2 bps, net round-trip maker cost of 4 bps vs 10 bps taker).
- **Queue Priority & Adverse Selection Modeling**: Implement queue position tracking and fill probability estimation. If the queue moves against the order or order book imbalance deteriorates, cancel and reprice immediately to mitigate adverse selection.

### R4. Institutional Purged Walk-Forward Evaluation & Live Telemetry Harness
- **Purged Walk-Forward Cross-Validation**: Evaluate all models across rolling out-of-sample windows with strict purging (dropping overlapping test return windows) and embargo (preventing auto-correlation leakage).
- **Clustered Run & Confidence Metrics**: Log every prediction and forward return at BBO mid-price. Group consecutive same-direction signals into independent runs, reporting:
  - Exact binomial p-values against $H_0: p = 0.50$.
  - Wilson 95% score confidence intervals.
  - Non-tie directional accuracy (zero-move ties scored as no-trade).
  - Drift-adjusted returns against secular benchmark drift.
  - Baselines: Always-Long, Always-Short, Random (10% signal rate), Single Best Member, Climatological Prior.
- **Continuous Ledger Flushing**: Stream all live signals, timestamps, BBO bid/ask/mid, fill events, and forward returns ($H1, H2, H3, H6, H12, H18$) to persistent disk ledgers.

---

## Acceptance Criteria

### AC1: Ensemble De-Correlation & Orthogonality
- [ ] Pairwise Pearson and Spearman correlation of raw directional predictions between any two ensemble members is $\le 0.70$ across the validation set.
- [ ] The ensemble integrates at least 3 distinct feature families (Cross-asset lead-lag, Spot-Perp basis/funding, Order book microstructure/queue imbalance).

### AC2: Probabilistic Calibration & Meta-Labeler Validity
- [ ] The meta-labeler produces continuous probability outputs across the full interval $[0, 1]$ with no artificial piecewise clamps or constant fallback buckets.
- [ ] Expected Calibration Error (ECE) is $\le 8.0\%$ on out-of-fold validation data.
- [ ] Brier score improves upon the climatological prior baseline ($< 0.2786$).

### AC3: Zero Leakage & Time-Series Integrity
- [ ] Formal audit passes verifying that 100% of feature inputs at bar $t$ use only information timestamped $\le T_{\text{close}} - 1000\text{ ms}$.
- [ ] The micro-signal / sentiment branch is verified to contain zero intra-bar or future price return leakage.

### AC4: Net Positive Profitability Under Maker Execution
- [ ] In purged walk-forward backtesting spanning $\ge 200$ independent signal clusters across multiple volatility regimes, the strategy achieves:
  - Positive net PnL after realistic Binance USD-M VIP0 maker fees (4.0 bps round-trip).
  - Wilson 95% confidence interval lower bound for directional accuracy $> 50.0\%$.
  - Lower average trade duration ($\le 30\text{ seconds}$) matching alpha decay.

### AC5: Automated Reproducible Test Suite & Live Harness
- [ ] An automated test suite runs end-to-end and validates data integrity, feature alignment, model inference latency ($\le 250\text{ ms}$), and execution logic.
- [ ] A unified evaluation script reproduces all metrics (Wilson CIs, p-values, ECE, net PnL, drift-adjusted alpha) directly from the telemetry ledger.

---

## Verification Commands & Harness
```powershell
# 1. Run Unit & Data Leakage Tests
pytest tests/test_leakage_and_features.py -v

# 2. Run Ensemble Orthogonality & Calibration Evaluation
python scripts/evaluate_ensemble_and_meta.py --data-dir "D:/backup/Binance data"

# 3. Run Purged Walk-Forward Simulation (Net of 4 bps Maker Fees)
python scripts/purged_walk_forward_backtest.py --fees-bps 4.0 --embargo-bars 3

# 4. Verify Telemetry & Ledger Rigor
python "feature engineering scripts/scratch/rigorous_evaluation_audit.py"
```


## 2026-10-18T02:24:38Z

# Teamwork Project Prompt — BTCUSDT 10-Second Alpha & Execution Overhaul (Continuation & Live Hardening)

> Status: Launched — Approved by User
> Goal: Multi-agent continuation of BTCUSDT 10s prediction & maker execution overhaul
> Target Team: Architecture, Quant ML Engineer, Execution Engine Specialist, Adversarial Auditor
> Integrity Mode: Development (Strict verification & validation)

## Project Overview & Objective
Continue the BTCUSDT 10-second-bar futures prediction and maker execution overhaul in `D:\backup\Binance data`. 

Milestones 1 through 4 have established initial implementations for:
1. **Orthogonal Features & De-correlated Ensemble**: 3 feature families (`cross_asset.py`, `basis_funding.py`, `microstructure_vpin.py`) with zero lookahead ($t \le T_{\text{close}} - 1000\text{ms}$) and pairwise correlation $|r| \le 0.2728$.
2. **Continuous Calibrated Meta-Labeler**: Unconstrained probability output via Platt scaling ($\text{ECE} \le 8.0\%$, $\text{Brier} < 0.2786$), $6.0\text{ bps}$ expected move hurdle, and realized volatility chop gate.
3. **BBO Queue-Aware Maker Execution Node**: Post-Only (`GTX`) limit orders at BBO, FIFO queue tracking ($Q_{\text{ahead}}$), adverse selection cancellation, dynamic ATR barriers, and 30s horizon.
4. **Purged Walk-Forward Harness**: 12 rolling purged folds ($\ge 200$ clusters, positive net PnL under 4.0 bps VIP0 maker fees).

The team's objective is to complete the hardening, adversarial audit, live testnet streaming integration, and end-to-end verification.

Working directory: `D:\backup\Binance data`

---

## Requirements

### R1. Adversarial Audit & Edge-Case Stress Testing
- Conduct rigorous multi-agent adversarial audit of all newly authored modules (`feature engineering scripts/features/`, `feature engineering scripts/meta/`, `feature engineering scripts/execution/`).
- Verify zero lookahead leakage, strict $1000\text{ ms}$ timestamp barriers, numerical stability under extreme market conditions (zero liquidity, inverted basis, zero volatility chop, missing bookTicker ticks), and contract compliance.

### R2. Live Ingestion & Streaming Loop Integration
- Wire the decoupled orthogonal feature pipeline and continuous meta-labeler directly into `live_loop_v2.py` and `binance_ws.py`.
- Ensure live incoming WebSocket messages from Binance USD-M (bookTicker, aggTrade) feed the feature pipeline strictly respecting $t \le T_{\text{close}} - 1000\text{ ms}$.
- Replace old hardcoded inference and heuristic blending in `live_loop_v2.py` with the calibrated ensemble and meta-labeler.

### R3. Autonomous BBO Maker Order Execution on Testnet
- Integrate `BBOExecutionNode` into the live testnet execution loop (`binance_testnet_engine.py`).
- Implement real-time order lifecycle management: place `GTX` limit orders at BBO, track queue position, cancel immediately upon adverse book deterioration, and enforce maximum 30s holding duration with dynamic ATR barriers.

### R4. Automated Continuous Telemetry & Ledger Logging
- Stream every live prediction, BBO quote, queue state, order intent, and fill outcome to a persistent disk ledger (`scratch/live_streaming_eval_ledger.jsonl`).
- Automatically resolve forward mid returns ($H1, H2, H3, H6, H12, H18$) with zero-tie tracking and Wilson confidence intervals.

---

## Acceptance Criteria

### AC1: Ensemble De-Correlation & Orthogonality
- [ ] Pairwise Pearson and Spearman correlation between any two ensemble members remains $\le 0.70$ across validation data.
- [ ] 3 distinct feature families remain fully decoupled without shared heuristic blending.

### AC2: Probabilistic Calibration & Hurdle Integrity
- [ ] Meta-labeler probability outputs are strictly continuous across $[0, 1]$ with zero artificial piecewise clamps.
- [ ] Expected Calibration Error (ECE) is $\le 8.0\%$ and Brier score $< 0.2786$.
- [ ] Trades are suppressed whenever expected move $< 6.0\text{ bps}$ or realized volatility indicates low-volatility chop.

### AC3: Zero Lookahead & Time-Series Integrity
- [ ] 100% of feature inputs at bar $t$ use only information timestamped $\le T_{\text{close}} - 1000\text{ ms}$.
- [ ] Live streaming loop passes data leakage audit with zero future or unclosed-bar information leakage.

### AC4: Net Positive Profitability Under Maker Execution
- [ ] Purged walk-forward simulation across $\ge 200$ independent signal clusters demonstrates:
  - Positive net PnL after realistic 4.0 bps round-trip maker fees.
  - Wilson 95% confidence interval lower bound $> 50.0\%$.
  - Average trade duration $\le 30\text{ seconds}$.

### AC5: End-to-End Test Suite & Latency
- [ ] Master test suite (`tests/test_leakage_and_features.py`) passes 100% (48/48 tests).
- [ ] Live inference latency per 10-second bar remains $\le 250\text{ ms}$.

---

## Verification Commands & Harness
```powershell
# 1. Master E2E & Contract Test Suite
pytest tests/test_leakage_and_features.py -v

# 2. Ensemble De-Correlation & Meta Calibration
python scripts/evaluate_ensemble_and_meta.py --data-dir "D:/backup/Binance data"

# 3. Purged Walk-Forward Simulation (Net of 4 bps Maker Fees)
python scripts/purged_walk_forward_backtest.py --fees-bps 4.0 --embargo-bars 3

# 4. Rigorous Live Ledger Audit & Telemetry
python "feature engineering scripts/scratch/rigorous_evaluation_audit.py"
```
