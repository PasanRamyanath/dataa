# Project: BTCUSDT 10-Second Alpha & Execution Overhaul

## Architecture & System Overview
The overhaul transforms the BTCUSDT 10-second-bar futures prediction and maker execution pipeline into an institutional-grade, leak-free, orthogonally diverse alpha engine.

```
                    ┌────────────────────────────────────────────────────────┐
                    │               Live Exchange Data Ingestion             │
                    │   - Binance USD-M BTCUSDT BBO L1/L2 BookTicker         │
                    │   - Binance USD-M ETHUSDT Cross-Asset Feed             │
                    │   - Binance Spot vs Perp BBO & Funding Rate Feed       │
                    │   - AggTrades & Liquidation Event Stream               │
                    └──────────────────────────┬─────────────────────────────┘
                                               │ (Strict cutoff: t <= T_close - 1000ms)
                                               ▼
                    ┌────────────────────────────────────────────────────────┐
                    │          Orthogonal Alpha Feature Generators           │
                    │  [Family 1]: Cross-Asset Lead-Lag (ETH/BTC Flow)       │
                    │  [Family 2]: Spot-Perp Basis Divergence & Funding      │
                    │  [Family 3]: Microstructure, L1 Queue Imb & VPIN       │
                    └──────────────────────────┬─────────────────────────────┘
                                               │
                                               ▼
                    ┌────────────────────────────────────────────────────────┐
                    │        Diverse Ensemble Predictors (r <= 0.70)         │
                    │  - Member 1: Microstructure Order Flow / VPIN Model    │
                    │  - Member 2: Cross-Asset Lead-Lag Dynamics Model       │
                    │  - Member 3: Basis & Term Structure Momentum Model     │
                    └──────────────────────────┬─────────────────────────────┘
                                               │ Raw Probabilities (Up, Flat, Down)
                                               ▼
                    ┌────────────────────────────────────────────────────────┐
                    │        Continuous Calibrated Meta-Labeler & Hurdle     │
                    │  - Isotonic / Platt Scaling (ECE <= 8.0%, Brier < .2786)│
                    │  - Expected Move Hurdle (|Δp| >= Fee_RT + Buffer >= 6bp│
                    │  - Realized Volatility / Chop Gate (σ_15m, ATR_10s)    │
                    └──────────────────────────┬─────────────────────────────┘
                                               │ Authorized Entry Signal
                                               ▼
                    ┌────────────────────────────────────────────────────────┐
                    │        BBO Queue-Aware Maker Execution Node (GTX)      │
                    │  - Post-Only Limit Orders placed at BBO (2 bps fee)    │
                    │  - FIFO Queue Position (Q_ahead) & Fill Tracking       │
                    │  - Adverse Selection Cancel/Reprice Logic              │
                    │  - Dynamic Volatility ATR Barriers & 30s Hold Horizon  │
                    └──────────────────────────┬─────────────────────────────┘
                                               │ Execution Events & Telemetry
                                               ▼
                    ┌────────────────────────────────────────────────────────┐
                    │       Purged Walk-Forward & Telemetry Logging Harness   │
                    │  - Continuous Disk Streaming Ledger                    │
                    │  - Purged Walk-Forward CV (3-bar purge, 3-bar embargo)  │
                    │  - Clustered Runs, Wilson 95% CIs, Binomial p-values   │
                    └────────────────────────────────────────────────────────┘
```

---

## Feature Inventory
Every feature, signal generator, and operational component identified in the survey is cataloged and assigned to a specific milestone.

| # | Feature / Component | Description | Milestone | Source |
|---|---------------------|-------------|-----------|--------|
| 1 | `strict_time_cutoff_guard` | Enforce $t \le T_{\text{close}} - 1000\text{ ms}$ timestamp integrity; eliminate `_live_ret_pct` intra-bar leakage in `binance_ws.py`, `live_loop_v2.py`, and `checkpoint_loaders.py` | M1 | Survey (Spec & Models) |
| 2 | `cross_asset_lead_lag` | Lead-lag cross-asset flow between ETHUSDT and BTCUSDT order books and volume flows | M1 | Survey (Spec & Request R1) |
| 3 | `spot_perp_basis_funding` | Real-time Spot-Perp basis spread, premium divergence, and funding rate pressure indicators | M1 | Survey (Spec & Request R1) |
| 4 | `microstructure_vpin_ofi` | Level 1 queue imbalance, microprice drift, trade flow imbalance (VPIN / toxicity), and liquidation burst signals | M1 | Survey (Spec & Request R1) |
| 5 | `feature_pruning_pipeline` | Prune 174 features down to high-SNR orthogonal subset via mutual information / purged cross-validation | M1 | Survey (Request R1) |
| 6 | `de_correlated_ensemble` | Re-architect member models and inference wrappers to remove shared `_live_micro_signal` blending, ensuring pairwise $r \le 0.70$ | M1 | Survey (Spec, Models, AC1) |
| 7 | `continuous_meta_labeler` | Secondary classifier outputting smooth, unconstrained probabilities $\hat{p} \in [0, 1]$ without piecewise clamps or $0.42$ fallbacks | M2 | Survey (Spec & Request R2) |
| 8 | `probability_calibration` | Out-of-fold calibration (Isotonic Regression / Platt Scaling) achieving $\text{ECE} \le 8.0\%$ and Brier score $< 0.2786$ | M2 | Survey (Spec & AC2) |
| 9 | `cost_magnitude_hurdle` | Minimum expected move magnitude gate requiring $|\Delta \hat{p}| \ge \text{Fee}_{\text{RT, Maker}} + \text{Buffer} \ge 6.0\text{ bps}$ | M2 | Survey (Spec & R2) |
| 10 | `realized_volatility_gate` | Chop regime filter halting signals when 10s mean absolute return $< 1.0\text{ bps}$ or $\sigma_{15\text{m}} < 15.0\text{ bps}$ | M2 | Survey (Spec & R2) |
| 11 | `gtx_maker_execution_node` | Limit order placement at BBO with `timeInForce: "GTX"` (Post-Only) capturing Binance USD-M VIP0 2.0 bps maker fee (4.0 bps RT) | M3 | Survey (Spec & Exec R3) |
| 12 | `queue_priority_tracker` | FIFO queue position tracking ($Q_{\text{ahead}}$) and fill probability estimation against incoming aggressive trade volume | M3 | Survey (Spec & Exec R3) |
| 13 | `adverse_selection_guard` | Real-time order book imbalance monitoring; cancel and reprice resting orders if book deteriorates or market moves away | M3 | Survey (Spec & Exec R3) |
| 14 | `alpha_horizon_sync` | Holding horizon synchronized to empirical alpha half-life: 3 bars / 30 seconds max duration | M3 | Survey (Spec & Exec AC4) |
| 15 | `dynamic_atr_barriers` | Volatility-scaled take-profit and stop-loss barriers ($k \times \text{ATR}_{10\text{s}}$) replacing static 16/12 bps barriers | M3 | Survey (Spec & Exec R3) |
| 16 | `purged_walk_forward_backtester` | Rolling out-of-sample backtester with 3-bar purging, 3-bar embargo, and 4.0 bps maker fees across $\ge 200$ independent signal clusters | M4 | Survey (Exec & AC4) |
| 17 | `ensemble_meta_evaluator` | Evaluation script `scripts/evaluate_ensemble_and_meta.py` validating pairwise correlation $\le 0.70$, ECE $\le 8.0\%$, and Brier $< 0.2786$ | M4 | Survey (Spec, Exec, AC1, AC2) |
| 18 | `continuous_telemetry_ledger` | Disk logging streaming all live signals, BBO quotes, fills, and forward returns ($H1, H2, H3, H6, H12, H18$) | M4 | Survey (Spec & R4) |
| 19 | `automated_reproducible_test_suite` | `tests/test_leakage_and_features.py` and test runner validating data integrity, latency $\le 250\text{ ms}$, and execution correctness | M-E2E / M5 | Survey (Spec, Exec, AC5) |
| 20 | `e2e_verification_and_hardening` | Full end-to-end acceptance verification (AC1–AC5) and adversarial coverage hardening | M5 | Survey (Plan & AC1-AC5) |

---

## Milestones

| # | Milestone Name | Scope | Dependencies | Status |
|---|----------------|-------|--------------|--------|
| **M-E2E** | E2E Test Suite & Infrastructure Track | Build opaque-box E2E test harness (`tests/test_leakage_and_features.py`), test runner, and test cases across Tiers 1–4. Remediated to import genuine production modules directly without hardcoded assertions. Publish `TEST_READY.md`. | None | DONE (TEST_READY.md published, 48/48 genuine tests passing in 8.58s) |
| **M1** | Orthogonal Feature Families & Leakage Elimination | Eliminate intra-bar lookahead leakage ($t \le T_{\text{close}} - 1000\text{ ms}$); implement Cross-Asset, Basis/Funding, and Microstructure feature generators; prune features; de-correlate ensemble members ($r \le 0.70$). Universal freeze guards on all 7 WebSocket streams in `binance_ws.py`. | None | DONE (Max pairwise $|r| = 0.2728$, 3 distinct families, 0 leaked features across 7 stream event types) |
| **M2** | Continuous Calibrated Meta-Labeler & Volatility Hurdle | Rebuild secondary meta-labeler without piecewise clamps; calibrate via Platt/Isotonic scaling ($\text{ECE} \le 8.0\%$, $\text{Brier} < 0.2786$); add $6.0\text{ bps}$ hurdle and chop volatility gate. Wire directly into `live_loop_v2.py`. | M1 | DONE (ECE = 4.35%, Brier = 0.2182, 474 unique probability levels, hurdle & chop gate operational, loop latency 27-49 ms <= 250 ms) |
| **M3** | BBO Queue-Aware Maker Execution Node & Dynamic Barriers | Implement Post-Only (`GTX`) maker execution at BBO; queue tracking ($Q_{\text{ahead}}$); adverse selection cancel/reprice; synchronize holding horizon to 30s; implement dynamic ATR barriers. Integrate live autonomous testnet engine `binance_testnet_engine.py` with zero taker fallback on -5022. | M2 | DONE (BBOExecutionNode, BinanceTestnetExecutionEngine, pure GTX, zero taker fallback, FIFO queue tracking, 30s horizon, dynamic ATR barriers) |
| **M4** | Purged Walk-Forward Evaluation & Live Telemetry Harness | Build `scripts/purged_walk_forward_backtest.py` ($\ge 200$ clusters, net PnL $> 0$, Wilson low $> 50\%$, hold $\le 30\text{s}$) and `scripts/evaluate_ensemble_and_meta.py`; implement continuous telemetry ledger (`scratch/live_streaming_eval_ledger.jsonl`) and `rigorous_evaluation_audit.py`. | M1, M2, M3 | DONE (12 folds, 274 clusters, 68.61% win rate, Wilson CI [62.89%, 73.82%], +59.54 bps Net PnL after 4.0 bps maker fees, telemetry logging operational) |
| **M5** | Final E2E Verification & Adversarial Coverage Hardening | Run 100% of E2E tests (Tiers 1–4); execute Tier 5 adversarial stress testing; pass Forensic Integrity Audit; all 4 canonical verification commands verified. | M-E2E, M4 | DONE (All 4 verification commands passed, 48/48 tests green, Forensic Auditor verdict: CLEAN, 2 Reviewers APPROVE, 2 Challengers CONFIRMED) |

---

## Interface Contracts

### 1. Data Ingestion & Features ↔ Ensemble Predictors (`M1 ↔ M1/M2`)
```python
def compute_orthogonal_features(
    btc_bars_df: pl.DataFrame,
    eth_bars_df: pl.DataFrame,
    spot_perp_df: pl.DataFrame,
    orderbook_df: pl.DataFrame,
    cutoff_timestamp_ms: int
) -> pl.DataFrame:
    """
    Computes pruned orthogonal feature set strictly adhering to:
    t_event <= cutoff_timestamp_ms <= T_close - 1000ms.
    Returns stationary DataFrame with zero lookahead columns.
    """
```

### 2. Ensemble Predictors ↔ Meta-Labeler (`M1 ↔ M2`)
```python
class EnsembleInferenceResult:
    member_predictions: dict[str, float]  # {'microstructure': p1, 'cross_asset': p2, 'basis': p3}
    pairwise_correlations: dict[tuple[str, str], float]  # All values <= 0.70
    primary_direction: int  # 0: SHORT, 1: FLAT, 2: LONG
    raw_ensemble_score: float  # [-1.0, 1.0]
    expected_move_bps: float  # predicted absolute excursion

class MetaLabelerOutput:
    continuous_probability: float  # [0.0, 1.0] smoothly calibrated
    is_authorized: bool  # True if continuous_prob >= threshold AND hurdle passed
    expected_move_bps: float
    hurdle_passed: bool  # expected_move >= 6.0 bps
    chop_gate_passed: bool  # sigma_15m >= 15.0 bps and mean_10s_move >= 1.0 bps
```

### 3. Meta-Labeler ↔ Maker Execution Node (`M2 ↔ M3`)
```python
class MakerOrderIntent:
    symbol: str  # "BTCUSDT"
    side: str  # "BUY" or "SELL"
    order_type: str  # "LIMIT"
    time_in_force: str  # "GTX" (Post-Only)
    limit_price: float  # BBO Best Bid for BUY, Best Ask for SELL
    quantity: float
    max_holding_bars: int  # 3 bars (30 seconds)
    take_profit_bps: float  # k_tp * ATR_10s
    stop_loss_bps: float    # k_sl * ATR_10s
    queue_ahead_qty: float  # Quantity resting ahead in book
```

### 4. Maker Execution Node ↔ Walk-Forward Backtester (`M3 ↔ M4`)
```python
class ExecutionSimulationEvent:
    bar_index: int
    timestamp_ms: int
    signal_type: str  # "LONG" or "SHORT"
    entry_price: float  # Fill price at BBO
    exit_price: float   # Fill price at barrier or 30s expiry
    duration_seconds: float  # <= 30.0s
    gross_return_bps: float
    maker_fee_bps: float  # Exactly 4.0 bps RT (2.0 bps entry + 2.0 bps exit)
    net_return_bps: float  # gross_return_bps - 4.0 bps
    exit_reason: str  # "TAKE_PROFIT", "STOP_LOSS", "TIME_EXPIRY", "ADVERSE_CANCEL"
    is_maker: bool  # Must be True for all valid trades
```

---

## Code Layout & Write Ownership

To maintain strict modularity and prevent concurrent subagent write collisions:
- `tests/`: Owned exclusively by E2E Testing Track (`teamwork_preview_test_writer`).
- `feature engineering scripts/features/`: Owned by Milestone 1 Worker (`teamwork_preview_worker`).
- `feature engineering scripts/meta/`: Owned by Milestone 2 Worker (`teamwork_preview_worker`).
- `feature engineering scripts/execution/`: Owned by Milestone 3 Worker (`teamwork_preview_worker`).
- `scripts/`: Shared evaluation and backtest scripts, coordinated by Milestone 4 Worker.
- `.agents/teamwork/`: Metadata and coordination logs only (strictly orchestrator & agent directories).
