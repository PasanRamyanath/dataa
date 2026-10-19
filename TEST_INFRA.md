# BTCUSDT 10-Second Alpha & Execution Overhaul — Test Infrastructure Specification (TEST_INFRA.md)

## 1. Test Philosophy & Principles

The testing framework for the BTCUSDT 10-Second Alpha & Execution Overhaul is architected under strict institutional quantitative engineering principles:

1. **Opaque-Box Requirement-Driven Testing**: Tests are formulated against external interface contracts, observable mathematical properties, and empirical execution telemetry. The test suite does not couple to transient internal implementation details or private helper methods; any refactoring that preserves contracts, mathematical correctness, and operational invariants will pass without modification.
2. **Strict Zero-Leakage & Temporal Causality Guarantee**: In high-frequency 10-second alpha generation, any lookahead leakage (even by a single millisecond) invalidates backtesting. The test suite strictly verifies:
   - Point-in-time timestamp integrity: $t_{\text{event}} \le T_{\text{close}} - 1000\text{ ms}$ for every feature input.
   - Micro-signal and sentiment purity: zero correlation with unclosed bar price changes or future returns.
   - Purged & embargoed split isolation: $T_{\text{train\_max}} < T_{\text{val\_min}}$ with guaranteed embargo intervals ($N_{\text{embargo}} \ge 3$ bars) and purging of overlapping forward target windows ($N_{\text{purge}} \ge 3$ bars).
3. **Probabilistic & Statistical Rigor**: Statistical claims must withstand rigorous hypothesis testing. Tests verify:
   - Exact two-sided binomial p-values ($H_0: p = 0.50$).
   - Wilson 95% score confidence intervals (ensuring lower bounds exceed $50.0\%$).
   - Proper Expected Calibration Error (ECE $\le 8.0\%$) and Brier score benchmark superiority ($< 0.2786$).
4. **Execution Realism & Microstructure Fidelity**: Maker execution is tested under realistic post-only constraints:
   - Maker fee tier: Binance USD-M VIP0 (2.0 bps entry + 2.0 bps exit = 4.0 bps round-trip).
   - Queue priority modeling ($Q_{\text{ahead}}$) and adverse selection cancel/reprice dynamics.
   - Empirical alpha decay horizon matching: maximum position holding duration is 3 bars / 30 seconds.
5. **Progressive Testability & Deterministic Reproducibility**: Tests are self-contained, isolated, and seed-controlled ($seed = 42$), allowing automated validation in CI/CD and offline development environments without external network dependencies.

---

## 2. Comprehensive Feature Inventory (20 Features)

Mapping every feature and component from `PROJECT.md` to its requirements, milestone, contract interface, and test tier coverage:

| # | Feature / Component ID | Milestone | Target Requirement | Contract / Module Interface | Tier 1 Unit Test Coverage | Tier 2/3/4 Test Coverage |
|---|------------------------|-----------|--------------------|-----------------------------|---------------------------|--------------------------|
| 1 | `strict_time_cutoff_guard` | M1 | R1, AC3 | `cutoff_timestamp_ms <= T_close - 1000ms` | Test cutoff enforcement, negative latency rejection, millisecond boundary | Tier 2: exact 1000ms boundary; Tier 3: Ingestion->Feature handoff |
| 2 | `cross_asset_lead_lag` | M1 | R1, AC1 | ETHUSDT flow & lead-lag basis | Test ETH order flow imbalance, lead-lag cross-correlation, ETH volume burst | Tier 2: zero ETH volume, NaN handling; Tier 4: Scenario 5 Lead-Lag shock |
| 3 | `spot_perp_basis_funding`| M1 | R1, AC1 | Spot-Perp basis spread & funding | Test basis calculation, funding rate pressure, premium divergence | Tier 2: negative basis, funding spike; Tier 3: Feature->Ensemble |
| 4 | `microstructure_vpin_ofi`| M1 | R1, AC1 | L1 queue imbalance, microprice, VPIN | Test VPIN calculation, OFI sign, microprice vs mid-price, liquidation burst | Tier 2: empty orderbook, zero trades; Tier 4: Scenario 2 Liquidity squeeze |
| 5 | `feature_pruning_pipeline`| M1 | R1 | Pruning 174 -> high-SNR orthogonal | Test stationarity check, zero lookahead exclusion, dead feature filtering | Tier 2: constant column pruning; Tier 3: Feature->Model contract |
| 6 | `de_correlated_ensemble` | M1 | R1, AC1 | Ensemble pairwise $r \le 0.70$ | Test Pearson correlation, Spearman rank correlation, diversity across 3 families | Tier 2: collinear input rejection; Tier 3: Member->Ensemble score aggregation |
| 7 | `continuous_meta_labeler`| M2 | R2, AC2 | Smooth $\hat{p} \in [0, 1]$ | Test continuous output range, absence of 0.42 clamp, monotonic sensitivity | Tier 2: extreme logit saturation; Tier 3: Ensemble->Meta handoff |
| 8 | `probability_calibration`| M2 | R2, AC2 | Isotonic/Platt, ECE $\le 8\%$, Brier | Test ECE calculation, reliability diagram binning, Brier score vs climatological | Tier 2: uniform/degenerate distributions; Tier 4: All Scenarios |
| 9 | `cost_magnitude_hurdle` | M2 | R2 | Move $\ge \text{Fee}_{\text{RT}} + \text{Buffer} \ge 6\text{bp}$ | Test hurdle thresholding, authorization logic, marginal pass/fail | Tier 2: exact 6.0 bps boundary; Tier 3: Meta->Execution handoff |
| 10| `realized_volatility_gate`| M2 | R2 | Chop filter: $\sigma_{15m} \ge 15\text{bp}$, $\mu_{10s} \ge 1\text{bp}$ | Test realized volatility calculation, mean 10s move filter, chop suppression | Tier 2: zero volatility regime; Tier 4: Scenario 1 Chop regime |
| 11| `gtx_maker_execution_node`| M3 | R3, AC4 | Limit order at BBO, `timeInForce: GTX` | Test BBO price selection, GTX post-only flag, 4.0 bps RT fee calculation | Tier 2: taker cross prevention; Tier 3: OrderIntent creation |
| 12| `queue_priority_tracker` | M3 | R3 | FIFO queue $Q_{\text{ahead}}$ tracking | Test queue position decrement, fill estimation against volume, queue exhaustion | Tier 2: zero book depth, infinite queue; Tier 4: Scenario 3 Queue depletion |
| 13| `adverse_selection_guard`| M3 | R3 | Cancel/reprice on book deterioration | Test imbalance shift detection, price move away detection, cancel signal | Tier 2: rapid quote flicker; Tier 4: Scenario 3 Adverse cancel-reprice |
| 14| `alpha_horizon_sync` | M3 | R3, AC4 | Max duration = 3 bars / 30 seconds | Test 30s timeout trigger, 3-bar holding limit, alpha decay alignment | Tier 2: boundary bar 3 vs bar 4; Tier 4: Scenario 4 Secular drift |
| 15| `dynamic_atr_barriers` | M3 | R3 | $k \times \text{ATR}_{10s}$ dynamic TP/SL | Test ATR scaling, dynamic barrier calculation, barrier touch event | Tier 2: ATR explosion, zero ATR fallback; Tier 3: Execution event logging |
| 16| `purged_walk_forward_backtester` | M4 | R4, AC4 | 3-bar purge, 3-bar embargo, $\ge 200$ clusters | Test purge split generation, embargo isolation, no-overlap assertion | Tier 2: minimum fold size error; Tier 4: Multi-fold backtest |
| 17| `ensemble_meta_evaluator` | M4 | R4, AC1, AC2 | Metrics: Wilson CI, p-val, ECE, Brier | Test Wilson CI score intervals, exact binomial test, drift-adjusted alpha | Tier 2: zero wins / 100% wins; Tier 3: Evaluator on mock ledger |
| 18| `continuous_telemetry_ledger` | M4 | R4 | Streaming signals, BBO, forward rets | Test JSONL serialization, forward horizon fields (H1-H18), disk flushing | Tier 2: disk error resilience, malformed record handling |
| 19| `automated_reproducible_test_suite`| M-E2E | AC5 | Pytest suite, latency $\le 250\text{ms}$ | Test execution time per inference, full suite pass, pytest compatibility | Tier 2: latency timeout assertions |
| 20| `e2e_verification_and_hardening` | M5 | AC1-AC5 | Acceptance criteria synthesis | Verify AC1, AC2, AC3, AC4, AC5 end-to-end against integrated pipeline | Tier 4: Full-pipeline multi-scenario stress |

---

## 3. Test Architecture & Structure

### Directory Layout
```
D:\backup\Binance data\
├── TEST_INFRA.md                   # This specification
├── TEST_READY.md                   # Final readiness report & runner instructions
├── tests/
│   ├── __init__.py
│   ├── contracts.py                # Pure interface contract dataclasses & schemas
│   ├── fixtures/
│   │   ├── __init__.py
│   │   └── synthetic_market.py     # Microsecond-accurate synthetic market data generator
│   └── test_leakage_and_features.py# Master test suite covering Tiers 1–4 across all 20 features
```

### Test Hierarchy (Tiers 1 to 4)
- **Tier 1: Unit & Contract Compliance**:
  - Independent unit verification for all 20 features.
  - Strict type and boundary assertions on inputs, intermediate calculations, and returned dataclasses.
  - Verification that outputs adhere strictly to the schemas defined in `PROJECT.md § Interface Contracts`.
- **Tier 2: Boundary, Edge & Adversarial Conditions**:
  - Null/empty DataFrames and zero-trade bars.
  - Millisecond timestamp boundaries ($T_{\text{close}} - 1000\text{ms}$ vs $T_{\text{close}} - 999\text{ms}$).
  - Collinear inputs and constant feature vectors.
  - Ultra-low volatility (zero ATR) and extreme volatility spikes.
  - Extreme queue sizes ($Q_{\text{ahead}} \to \infty$ and $Q_{\text{ahead}} = 0$).
  - Negative basis spreads and funding spikes.
- **Tier 3: Pairwise Cross-Feature Interactions**:
  - Boundary 1: `strict_time_cutoff_guard` $\leftrightarrow$ `compute_orthogonal_features`
  - Boundary 2: Feature generators $\leftrightarrow$ `de_correlated_ensemble` ($r \le 0.70$)
  - Boundary 3: Ensemble outputs $\leftrightarrow$ `continuous_meta_labeler` & `cost_magnitude_hurdle`
  - Boundary 4: Meta-labeler authorization $\leftrightarrow$ `gtx_maker_execution_node` (`MakerOrderIntent`)
  - Boundary 5: Maker execution events $\leftrightarrow$ `purged_walk_forward_backtester` & `continuous_telemetry_ledger`
- **Tier 4: Real-World Application Scenarios**:
  - End-to-end multi-bar simulations modeling authentic exchange dynamics under historical regimes identified during the audit.

---

## 4. Real-World Application Scenarios (Tier 4)

### Scenario 1: Volatility Compression & Chop Regime (False Breakout Suppression)
- **Market Dynamics**: Market trades in a narrow consolidation band ($< 0.5\text{ bps}$ per 10s bar, 15m realized volatility $\sigma_{15m} \approx 8.0\text{ bps} < 15.0\text{ bps}$).
- **Observable Inputs**: Repeated micro-fluctuations crossing mid-price; low volume; order book tightly packed.
- **Expected System Behavior**: The `realized_volatility_gate` triggers. Even if raw ensemble outputs emit directional bias, `chop_gate_passed` is set to `False` and `is_authorized` is `False`. Zero maker orders are posted, preventing fee drag and choppy whipsaws.

### Scenario 2: Flash Liquidity Squeeze & Microstructure Cascade
- **Market Dynamics**: Aggressive market sell orders rapidly consume bids across multiple levels; Level 1 queue imbalance drops from $0.5$ to $0.05$; VPIN spikes above 90th percentile; large liquidation burst arrives.
- **Observable Inputs**: Level 1 bid quantity collapses; volume delta heavily negative; microprice shifts significantly below BBO mid.
- **Expected System Behavior**: Microstructure feature generator signals strong downward toxicity; member model 1 outputs short probability $\ge 0.80$; ensemble authorizes SHORT intent; maker order posts at Best Ask (post-only GTX); execution node tracks fill within alpha decay horizon (30s) capturing downward excursion.

### Scenario 3: Adverse Queue Depletion & Immediate Cancel-Reprice
- **Market Dynamics**: Maker buy order posted at Best Bid ($Q_{\text{ahead}} = 2.5\text{ BTC}$). As market evolves, large sell volume hits the book, ask side thickens, and bid depth thins dramatically ($OFI \ll 0$).
- **Observable Inputs**: Order book imbalance deteriorates past safety threshold ($> 30\%$ drop in supporting bid depth) while order remains resting in queue.
- **Expected System Behavior**: `adverse_selection_guard` detects adverse book transition before order fills; emits immediate `CANCEL` event; order is canceled without filling into informed toxic flow; prevents adverse selection loss.

### Scenario 4: Strong Secular Upward Drift with Alpha Horizon Decay
- **Market Dynamics**: Secular upward trend (+25.0 bps over 45 minutes), interspersed with minor pullbacks.
- **Observable Inputs**: Consistent positive drift; short signals occasionally generated by mean-reverting indicators.
- **Expected System Behavior**: Holding horizon is strictly locked to $\le 3\text{ bars}$ (30 seconds) per `alpha_horizon_sync`. Dynamic ATR barriers trigger profit-taking or time expiry at 30 seconds rather than lingering for 180 seconds into secular drift. Drift-adjusted evaluation confirms strategy edge separates from market trend.

### Scenario 5: Cross-Asset Lead-Lag Shock Propagation (ETH to BTC)
- **Market Dynamics**: Sudden 30 bps upward spike in ETHUSDT perpetual volume and price preceding BTCUSDT by 10–20 seconds.
- **Observable Inputs**: ETH order book imbalance and return spike arrive at $t \le T_{\text{close}} - 1000\text{ ms}$; BTC price has not yet moved.
- **Expected System Behavior**: `cross_asset_lead_lag` feature generator captures positive lead-lag indicator; ensemble member 2 weights the cross-asset signal; meta-labeler verifies expected move exceeds hurdle ($\ge 6.0\text{ bps}$); enters long maker order; executes before BTC catches up.

---

## 5. Quantitative Acceptance Criteria & Pass/Fail Thresholds

| Metric / Requirement | Acceptance Threshold | Failure Condition | Verification Method |
|----------------------|----------------------|-------------------|---------------------|
| Time Cutoff Offset | $\ge 1000\text{ ms}$ before $T_{\text{close}}$ | $t_{\text{event}} > T_{\text{close}} - 1000\text{ ms}$ | Timestamp differential check |
| Future Return Leakage | Pearson $r = 0.0000$ with $R_{t+k}$ | $|r| > 10^{-6}$ | Correlation against forward returns |
| Inference Latency | $\le 250\text{ ms}$ per bar | Latency $> 250\text{ ms}$ | Timing profiler over 100 bars |
| Ensemble Pairwise Correlation | $\le 0.70$ (Pearson & Spearman) | Pairwise $r > 0.70$ | Matrix correlation check across members |
| Ensemble Feature Diversity | $\ge 3$ distinct families | $< 3$ families | Family category tag audit |
| Meta-Labeler Continuity | $\hat{p} \in [0, 1]$, continuous | Discrete steps, constant 0.42 | Derivative / unique value check |
| Calibration ECE | $\le 8.0\%$ | $\text{ECE} > 8.0\%$ | Out-of-fold binned reliability error |
| Calibration Brier Score | $< 0.2786$ (beats climatological) | Brier $\ge 0.2786$ | Mean squared probabilistic error |
| Expected Move Hurdle | $|\Delta \hat{p}| \ge 6.0\text{ bps}$ | Authorized with move $< 6.0\text{ bps}$ | Hurdle gate assertion |
| Realized Volatility Chop Gate | $\mu_{10s} \ge 1.0\text{ bp}$, $\sigma_{15m} \ge 15.0\text{ bp}$ | Signal passed in chop | Gate status assertion |
| Maker Post-Only Order | `time_in_force == "GTX"` | Taker execution / non-GTX | Execution order type assertion |
| Net Maker Fee Modeling | Exactly $4.0\text{ bps}$ round-trip | Fee $\ne 4.0\text{ bps}$ | Transaction cost deduction check |
| Max Holding Horizon | $\le 3\text{ bars}$ / $30.0\text{ seconds}$ | Duration $> 30.0\text{ s}$ | Trade lifecycle duration check |
| Dynamic ATR Barriers | $k \times \text{ATR}_{10s}$ | Static 16/12 bps | Volatility scaling ratio check |
| Purged Walk-Forward Splits | $N_{\text{purge}} \ge 3$, $N_{\text{embargo}} \ge 3$ | $T_{\text{train\_max}} \ge T_{\text{val\_min}}$ | Split index temporal gap assertion |
| Wilson 95% CI Lower Bound | $> 50.0\%$ directional accuracy | CI Lower Bound $\le 50.0\%$ | Wilson score formula verification |

---

## 6. Execution Instructions

The complete test suite is executed using standard pytest:
```powershell
pytest tests/test_leakage_and_features.py -v
```
All tests must execute synchronously, produce zero warnings/errors, and complete in $< 30\text{ seconds}$.
