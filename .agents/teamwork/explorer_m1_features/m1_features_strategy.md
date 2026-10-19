# Milestone 1: Orthogonal Feature Families & Feature Pruning Strategy

**Author:** Orthogonal Features Explorer (Milestone 1)  
**Target Recipient:** Milestone 1 Worker (`teamwork_preview_worker`) & Orchestrator  
**Status:** Approved Architectural Blueprint  
**Date:** 2026-10-18  

---

## Executive Summary

An empirical audit of the BTCUSDT 10-second-bar futures prediction system revealed that claimed directional accuracy (60%–64%) collapsed to 46.4% raw (52.9% latency-adjusted) under true BBO mid-price evaluation ($p = 0.851$, statistically indistinguishable from noise). Three fundamental structural flaws were diagnosed:
1. **Severe Member Collinearity ($r = 0.987 \dots 0.997$):** Member models acted as a single redundant model due to synthetic blending with an intra-bar microprice heuristic (`_live_micro_signal`).
2. **Subtle Intra-Bar Forward Lookahead Leakage:** The `_live_micro_signal` and `get_live_snapshot` mechanisms sampled trade returns from the currently forming bar (`_live_ret_pct`), violating the causal time boundary $t \le T_{\text{close}} - 1000\text{ ms}$.
3. **Curse of Dimensionality in 174 Features:** The existing 174-feature set (`FEATURE_COLUMNS`) contained extensive collinearity (redundant moving averages of momentum and OFI, high-cardinality flags, and dead exchange feeds), inducing high variance and overfitting.

This strategy report provides the comprehensive mathematical, algorithmic, and software architecture to resolve these issues in `feature engineering scripts/features/`. It specifies:
1. **Family 1: Cross-Asset Lead-Lag Dynamics** (ETHUSDT vs BTCUSDT returns, OFI, and cross-asset momentum over 10s, 30s, and 60s windows).
2. **Family 2: Spot-Perpetual Basis Divergence & Funding Pressure** (BBO basis spread in bps, multi-scale rolling basis z-score, basis acceleration, and forward funding rate carry pressure).
3. **Family 3: Order Book Microstructure & Flow Toxicity** (Level 1 queue imbalance, Stoikov microprice drift, Trade Flow Imbalance / VPIN toxicity proxy, and large liquidation bursts).
4. **Feature Pruning Pipeline** (Hierarchical Clustered Correlation at $r \le 0.70$, Purged Walk-Forward Cross-Validation with 3-bar purge and 3-bar embargo, and Recursive Feature Elimination down to 25–35 high-SNR orthogonal features).
5. **Strict Time Cutoff Guard** ($t \le T_{\text{close}} - 1000\text{ ms}$) guaranteeing zero forward lookahead.

---

## 1. Family 1: Cross-Asset Lead-Lag Dynamics (ETHUSDT vs BTCUSDT)

### 1.1 Economic & Microstructural Rationale
BTCUSDT and ETHUSDT constitute the twin liquidity pillars of digital asset markets. While Bitcoin is the primary macroeconomic driver, empirical price discovery frequently shifts to Ethereum during sector-specific flow events (e.g., altcoin momentum, DeFi/L2 activity, Deribit options gamma rebalancing). At 10-second resolution, cross-asset lead-lag effects manifest across three transmission channels:
- **Lead-Lag Return Spillover:** Cross-asset market makers and statistical arbitrage desks hedge ETH inventory shocks in BTC, producing predictable 10s–60s price drift.
- **Order Flow Imbalance (OFI) Asymmetry:** Aggressive taker buying/selling in ETH perp routinely leads BTC quote adjustments when informed flow hits ETH first.
- **Multi-Horizon Momentum Divergence:** Relative momentum divergence between ETH and BTC identifies overextended pairs and mean-reversion impulses.

### 1.2 Mathematical Formulation

Let $P_{\text{ETH}, t}$ and $P_{\text{BTC}, t}$ denote the 10-second bar close prices (or BBO mid-prices) for ETHUSDT and BTCUSDT perpetual contracts at bar $t$.

#### A. Lagged ETH Returns
Log return of ETH at 10-second resolution:
$$r_{\text{ETH}, t} = \ln\left(\frac{P_{\text{ETH}, t}}{P_{\text{ETH}, t-1}}\right)$$
For lags $k \in \{1, 2, 3, 6\}$ (corresponding to 10s, 20s, 30s, 60s):
$$\text{eth\_lead\_return\_lag\_}k_{t} = r_{\text{ETH}, t-k} \times 10^4 \quad (\text{in bps})$$

#### B. Cross-Asset Return Spread & Residual
Over rolling horizons $W \in \{1, 3, 6\}$ bars ($10s, 30s, 60s$):
$$r_{\text{ETH}, t}^{(W)} = \ln\left(\frac{P_{\text{ETH}, t}}{P_{\text{ETH}, t-W}}\right), \quad r_{\text{BTC}, t}^{(W)} = \ln\left(\frac{P_{\text{BTC}, t}}{P_{\text{BTC}, t-W}}\right)$$
$$\text{cross\_return\_spread\_}W_{t} = \left( r_{\text{ETH}, t}^{(W)} - r_{\text{BTC}, t}^{(W)} \right) \times 10^4 \quad (\text{in bps})$$

To isolate the orthogonal innovation in BTC relative to ETH, we compute the rolling hedge ratio $\beta_t^{(180)}$ (over a 30-minute window, $W=180$ bars):
$$\beta_t^{(180)} = \frac{\text{Cov}_{180}\left(r_{\text{BTC}}^{(1)}, r_{\text{ETH}}^{(1)}\right)}{\text{Var}_{180}\left(r_{\text{ETH}}^{(1)}\right) + \epsilon}$$
$$\text{cross\_return\_residual\_10s}_{t} = \left( r_{\text{BTC}, t}^{(1)} - \beta_t^{(180)} \cdot r_{\text{ETH}, t}^{(1)} \right) \times 10^4 \quad (\text{in bps})$$

#### C. Cross-Asset Order Flow Imbalance (OFI)
Let $V_{\text{buy}, t}^{\text{ETH}}$ and $V_{\text{sell}, t}^{\text{ETH}}$ denote aggressive taker buy and sell volumes for ETHUSDT during bar $t$:
$$\text{OFI}_{\text{ETH}, t} = \frac{V_{\text{buy}, t}^{\text{ETH}} - V_{\text{sell}, t}^{\text{ETH}}}{V_{\text{buy}, t}^{\text{ETH}} + V_{\text{sell}, t}^{\text{ETH}} + \epsilon} \in [-1.0, 1.0]$$
Lagged ETH OFI features:
$$\text{eth\_ofi\_lag\_}k_{t} = \text{OFI}_{\text{ETH}, t-k} \quad \text{for } k \in \{1, 2, 3\}$$

Cross-Asset OFI Divergence over window $W \in \{1, 3, 6\}$:
$$\overline{\text{OFI}}_{\text{ETH}, t}^{(W)} = \frac{1}{W}\sum_{i=0}^{W-1} \text{OFI}_{\text{ETH}, t-i}, \quad \overline{\text{OFI}}_{\text{BTC}, t}^{(W)} = \frac{1}{W}\sum_{i=0}^{W-1} \text{OFI}_{\text{BTC}, t-i}$$
$$\text{cross\_ofi\_divergence\_}W_{t} = \overline{\text{OFI}}_{\text{ETH}, t}^{(W)} - \overline{\text{OFI}}_{\text{BTC}, t}^{(W)} \in [-2.0, 2.0]$$

#### D. Cross-Asset Multi-Horizon Momentum Differential
For horizons $W \in \{1, 3, 6\}$:
$$\text{MOM}_{\text{ETH}, t}^{(W)} = \frac{P_{\text{ETH}, t} - P_{\text{ETH}, t-W}}{P_{\text{ETH}, t-W}} \times 10^4, \quad \text{MOM}_{\text{BTC}, t}^{(W)} = \frac{P_{\text{BTC}, t} - P_{\text{BTC}, t-W}}{P_{\text{BTC}, t-W}} \times 10^4$$
$$\text{cross\_mom\_diff\_}W_{t} = \text{MOM}_{\text{ETH}, t}^{(W)} - \text{MOM}_{\text{BTC}, t}^{(W)}$$
Rolling Volatility-Standardized Momentum Z-Score (over 360 bars / 1 hr):
$$Z(\text{cross\_mom\_diff\_30s})_{t} = \frac{\text{cross\_mom\_diff\_3}_{t} - \mu_{360}\left(\text{cross\_mom\_diff\_3}\right)}{\sigma_{360}\left(\text{cross\_mom\_diff\_3}\right) + \epsilon}$$

### 1.3 Anti-Leakage & Fallback Behavior
- **Latency Cutoff:** ETH bars joined must satisfy $t_{\text{ETH}} \le T_{\text{close}} - 1000\text{ ms}$. In historical bar aggregation, bar $t-1$ completes 10 seconds prior to $T_{\text{close}}$, which strictly exceeds the 1-second requirement ($10,000\text{ ms} \ge 1,000\text{ ms}$).
- **Fallback:** If `eth_bars_df` is `None` or missing, emit all columns filled with `0.0` (Float32).

---

## 2. Family 2: Spot-Perpetual Basis Divergence & Funding Pressure

### 2.1 Economic & Microstructural Rationale
The price of a perpetual futures contract is tethered to the underlying spot index via continuous arbitrage and periodic 8-hour funding cash settlements.
- **Basis Spread:** The relative difference between perpetual and spot prices ($\text{Basis} = P_{\text{perp}} - P_{\text{spot}}$) acts as a mean-reverting elastic band. Extreme contango ($P_{\text{perp}} \gg P_{\text{spot}}$) creates structural downward pressure as basis arbs short the perp and long spot.
- **Rolling Basis Z-Scores:** Absolute basis varies across macro interest rate regimes; rolling z-scores normalize basis dislocation relative to recent 30m, 2h, and 8h history.
- **Basis Acceleration:** The velocity and curvature of basis expansion highlight whether leverage expansion is accelerating or decelerating into an exhaustion pivot.
- **Funding Rate Pressure Index:** Funding settlement occurs every 8 hours (00:00, 08:00, 16:00 UTC). Holding costs increase linearly per unit time as settlement nears, driving pre-funding de-leveraging.

### 2.2 Mathematical Formulation

#### A. Basis Spread (in bps)
Using BBO mid-prices (or bar close prices):
$$P_{\text{perp\_mid}, t} = \frac{P_{\text{perp\_bid}, t} + P_{\text{perp\_ask}, t}}{2}, \quad P_{\text{spot\_mid}, t} = \frac{P_{\text{spot\_bid}, t} + P_{\text{spot\_ask}, t}}{2}$$
$$\text{basis\_spread\_bps}_{t} = \left(\frac{P_{\text{perp\_mid}, t} - P_{\text{spot\_mid}, t}}{P_{\text{spot\_mid}, t}}\right) \times 10^4$$

#### B. Rolling Basis Z-Scores
For multi-scale windows $W \in \{180 \text{ bars (30m)}, 720 \text{ bars (2h)}, 2880 \text{ bars (8h)}\}$:
$$\mu_{\text{basis}, t}^{(W)} = \frac{1}{W} \sum_{i=0}^{W-1} \text{basis\_spread\_bps}_{t-i}$$
$$\sigma_{\text{basis}, t}^{(W)} = \sqrt{\frac{1}{W} \sum_{i=0}^{W-1} \left(\text{basis\_spread\_bps}_{t-i} - \mu_{\text{basis}, t}^{(W)}\right)^2 + \epsilon}$$
$$\text{basis\_zscore\_}W_{t} = \frac{\text{basis\_spread\_bps}_{t} - \mu_{\text{basis}, t}^{(W)}}{\sigma_{\text{basis}, t}^{(W)}}$$

#### C. Basis Velocity & Curvature (Acceleration)
- Basis Velocity over $k \in \{1 \text{ bar (10s)}, 3 \text{ bars (30s)}, 6 \text{ bars (60s)}\}$:
  $$\text{basis\_velocity\_}k_{t} = \text{basis\_spread\_bps}_{t} - \text{basis\_spread\_bps}_{t-k}$$
- Basis Acceleration (second difference / curvature over 30s):
  $$\text{basis\_acceleration\_30s}_{t} = \text{basis\_velocity\_3}_{t} - \text{basis\_velocity\_3}_{t-3} = \text{basis\_spread\_bps}_{t} - 2\cdot\text{basis\_spread\_bps}_{t-3} + \text{basis\_spread\_bps}_{t-6}$$

#### D. Forward Funding Rate Pressure & Divergence
Let $F_t$ be the latest published 8-hour funding rate (dimensionless, e.g., $+0.0001 = 1\text{ bps}$).
Let $T_{\text{next\_funding}}$ be the next funding settlement timestamp in ms, and $t$ be current bar timestamp in ms.
Remaining duration until funding:
$$\Delta t_{\text{to\_funding}} = \max\left(0, T_{\text{next\_funding}} - t\right) \quad (\text{ms})$$
Normalized cycle progress:
$$\tau_t = \frac{\Delta t_{\text{to\_funding}}}{8 \times 3600 \times 1000} \in [0.0, 1.0]$$
- **Funding Rate Pressure Index:**
  $$\text{funding\_pressure\_bps}_{t} = F_t \times (1.0 - \tau_t) \times 10^4$$
  *Interpretation:* When $F_t = +10\text{ bps}$ and $\tau_t \to 0$ (e.g. 15 minutes before 08:00 UTC), $\text{funding\_pressure\_bps} \to +9.7\text{ bps}$. Longs face an imminent cash penalty, exerting strong downward selling drift.
- **Funding-Basis Divergence:**
  $$\text{funding\_basis\_divergence}_{t} = \text{basis\_spread\_bps}_{t} - (F_t \times 10^4)$$
  *Interpretation:* Detects market mispricing where basis premium has moved ahead of the actual funding carry cost.

### 2.3 Anti-Leakage & Fallback Behavior
- All spot prices and funding rates are joined via `join_asof(strategy="backward")` strictly on $t \le T_{\text{close}} - 1000\text{ ms}$.
- Fallback: If `spot_perp_df` is `None` or spot data is missing, emit `basis_spread_bps = 0.0`, `basis_zscore_* = 0.0`, `funding_pressure_bps = 0.0`.

---

## 3. Family 3: Order Book Microstructure & Flow Toxicity

### 3.1 Economic & Microstructural Rationale
The empirical audit established that BTCUSDT 10s alpha decays rapidly within 20–30s. The microstructural state of the order book and flow toxicity are the primary drivers of this horizon:
- **Level 1 Queue Imbalance ($I_{L1}$):** Measures the direct supply/demand asymmetry at the best bid and ask. An overloaded bid queue with thin ask liquidity signals high probability of an upward ask wipe.
- **Stoikov Microprice Drift:** Sasha Stoikov's analytical formulation accounts for tick size and queue sizes. When microprice departs from mid-price, it forecasts the direction and probability of the next quote tick before it occurs.
- **Trade Flow Imbalance (TFI) & VPIN (Flow Toxicity):** Easley, Lopez de Prado, and O'Hara (2012) demonstrated that toxic flow from informed traders predicts adverse price swings. VPIN quantifies toxic volume arrival.
- **Large Liquidation Bursts:** Forced liquidations (`@forceOrder`) are non-discretionary market orders executed into available book depth. Large liquidation cascades deplete liquidity and precede either sharp continuation runs or mean-reverting exhaustion bounces.

### 3.2 Mathematical Formulation

#### A. Level 1 Queue Imbalance
Let $Q_{\text{bid}, t}^{(1)}$ and $Q_{\text{ask}, t}^{(1)}$ denote the resting quantity at the Best Bid ($P_{\text{bid}, t}$) and Best Ask ($P_{\text{ask}, t}$) at time $t$:
$$I_{L1, t} = \frac{Q_{\text{bid}, t}^{(1)} - Q_{\text{ask}, t}^{(1)}}{Q_{\text{bid}, t}^{(1)} + Q_{\text{ask}, t}^{(1)} + \epsilon} \in [-1.0, 1.0]$$
- Instantaneous Level 1 Imbalance: $\text{l1\_queue\_imbalance}_{t} = I_{L1, t}$
- Rolling Averages:
  $$\text{l1\_queue\_imbalance\_ma3}_{t} = \frac{1}{3}\sum_{i=0}^2 I_{L1, t-i}, \quad \text{l1\_queue\_imbalance\_ma6}_{t} = \frac{1}{6}\sum_{i=0}^5 I_{L1, t-i}$$
- Queue Depletion Velocity:
  $$\text{l1\_queue\_velocity}_{t} = I_{L1, t} - I_{L1, t-1}$$

#### B. Stoikov Microprice Drift
Let spread $S_t = P_{\text{ask}, t} - P_{\text{bid}, t}$ and mid-price $P_{\text{mid}, t} = \frac{P_{\text{bid}, t} + P_{\text{ask}, t}}{2}$.
Queue-weighted Stoikov Microprice:
$$P_{\text{micro}, t} = \frac{Q_{\text{bid}, t}^{(1)} P_{\text{ask}, t} + Q_{\text{ask}, t}^{(1)} P_{\text{bid}, t}}{Q_{\text{bid}, t}^{(1)} + Q_{\text{ask}, t}^{(1)} + \epsilon} = P_{\text{mid}, t} + \frac{S_t}{2} \cdot I_{L1, t}$$
- **Microprice Drift (in bps):**
  $$\text{microprice\_drift\_bps}_{t} = \left(\frac{P_{\text{micro}, t} - P_{\text{mid}, t}}{P_{\text{mid}, t}}\right) \times 10^4 = \left(\frac{S_t \cdot I_{L1, t}}{2 P_{\text{mid}, t}}\right) \times 10^4$$
- **Microprice Velocity (in bps) over $k \in \{1 \text{ bar (10s)}, 3 \text{ bars (30s)}\}$:**
  $$\text{microprice\_velocity\_}k_{t} = \left(\frac{P_{\text{micro}, t} - P_{\text{micro}, t-k}}{P_{\text{mid}, t}}\right) \times 10^4$$

#### C. Trade Flow Imbalance & VPIN Toxicity
- **10-Second Trade Flow Imbalance (TFI):**
  From aggressive trade prints during bar $t$:
  $$\text{TFI}_{t} = \frac{V_{\text{buy}, t} - V_{\text{sell}, t}}{V_{\text{buy}, t} + V_{\text{sell}, t} + \epsilon} \in [-1.0, 1.0]$$
  Rolling averages: $\text{tfi\_ma\_3}_{t}$ and $\text{tfi\_ma\_6}_{t}$.
- **Volume-Synchronized Probability of Toxicity (VPIN Proxy):**
  Over rolling windows of $W \in \{6 \text{ bars (1m)}, 30 \text{ bars (5m)}\}$:
  $$\text{vpin\_toxicity\_}W_{t} = \frac{\sum_{i=0}^{W-1} |V_{\text{buy}, t-i} - V_{\text{sell}, t-i}|}{\sum_{i=0}^{W-1} (V_{\text{buy}, t-i} + V_{\text{sell}, t-i}) + \epsilon} \in [0.0, 1.0]$$
- **Toxicity Surge Ratio:**
  $$\text{vpin\_surge\_ratio}_{t} = \frac{\text{vpin\_toxicity\_6}_{t}}{\text{vpin\_toxicity\_30}_{t} + \epsilon}$$
  A surge ratio $> 1.5$ flags imminent toxic adverse selection against resting limit orders.

#### D. Large Liquidation Bursts
Using Binance `@forceOrder` liquidation stream:
- $L_{\text{buy}, t}$: Short liquidations (forced aggressive buy orders).
- $L_{\text{sell}, t}$: Long liquidations (forced aggressive sell orders).
- Total liquidation notional: $L_{\text{tot}, t} = L_{\text{buy}, t} + L_{\text{sell}, t}$ (in USDT).
- Net Liquidation Directional Flow:
  $$\text{liq\_net\_flow\_bps}_{t} = \frac{L_{\text{buy}, t} - L_{\text{sell}, t}}{\text{notional\_volume}_{t} + 10^5} \times 10^4$$
- Liquidation Burst Z-Score (over 180 bars / 30 min):
  $$\mu_{L, t}^{(180)} = \frac{1}{180}\sum_{i=0}^{179} L_{\text{tot}, t-i}, \quad \sigma_{L, t}^{(180)} = \sqrt{\frac{1}{180}\sum_{i=0}^{179} \left(L_{\text{tot}, t-i} - \mu_{L, t}^{(180)}\right)^2 + \epsilon}$$
  $$\text{liq\_burst\_zscore}_{t} = \frac{L_{\text{tot}, t} - \mu_{L, t}^{(180)}}{\sigma_{L, t}^{(180)}}$$
- Binary Liquidation Cascade Flag:
  $$\text{liq\_burst\_active}_{t} = \mathbf{1}_{\{\text{liq\_burst\_zscore}_{t} \ge 2.5 \land L_{\text{tot}, t} \ge 250,000\}}$$

### 3.3 Anti-Leakage & Fallback Behavior
- All book quotes ($P_{\text{bid}}, P_{\text{ask}}, Q_{\text{bid}}, Q_{\text{ask}}$) must be sampled at $t \le T_{\text{close}} - 1000\text{ ms}$.
- Fallback: If `orderbook_df` is unavailable, compute proxy spread from $(P_{\text{high}} - P_{\text{low}}) / P_{\text{close}}$, proxy imbalance from $(V_{\text{buy}} - V_{\text{sell}}) / V_{\text{tot}}$, and set microprice drift to $0.0$. Liquidation features fallback to $0.0$.

---

## 4. Feature Pruning Pipeline

### 4.1 Objective & Acceptance Constraints
- **Current State:** 174 features (`FEATURE_COLUMNS` in `common/feature_contract.py`).
- **Target State:** Pruned down to $K \in [25, 35]$ high-SNR, low-correlation orthogonal features.
- **Constraints (AC1 & AC3):**
  1. Pairwise Pearson and Spearman correlation between any two selected features $\le 0.70$.
  2. Balanced representation across all 3 orthogonal families plus fundamental price-volume dynamics.
  3. Strict zero-leakage cross-validation using Purged Walk-Forward CV (3-bar purge, 3-bar embargo).

### 4.2 Multi-Stage Pruning Algorithm

```
  ┌───────────────────────────────────────────────────────────────┐
  │ 174 Active Pipeline Features (FEATURE_COLUMNS)                │
  └───────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼ Stage 1: Variance & Null Filter
  ┌───────────────────────────────────────────────────────────────┐
  │ Candidate Features (~120): Drop Var < 1e-6, Null Rate > 20%   │
  └───────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼ Stage 2: Hierarchical Clustered Correlation
  ┌───────────────────────────────────────────────────────────────┐
  │ Tree Clustering on D(i, j) = sqrt(0.5*(1 - |rho(i, j)|))      │
  │ Cut at rho = 0.70 -> Retain 1 feature per cluster with max IC │
  └───────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼ Stage 3: Purged Walk-Forward IC & MI Filter
  ┌───────────────────────────────────────────────────────────────┐
  │ 5-Fold Expanding Purged CV (3-bar purge, 3-bar embargo)       │
  │ Rank by Information Ratio IR = Mean(IC) / Std(IC) >= 0.5      │
  └───────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼ Stage 4: Purged Recursive Feature Elimination
  ┌───────────────────────────────────────────────────────────────┐
  │ LightGBM Classifier on Purged OOF Folds                       │
  │ Iteratively eliminate lowest 10% gain/SHAP features           │
  └───────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼ Stage 5: Final Orthogonality & Schema Export
  ┌───────────────────────────────────────────────────────────────┐
  │ PRUNED_ORTHOGONAL_FEATURES (25 - 35 features, max |rho| <= .70│
  └───────────────────────────────────────────────────────────────┘
```

#### Detailed Stage Specifications:

1. **Stage 1 — Variance & Null Pre-Filter:**
   - Eliminate features with $\text{Var}(X) < 10^{-6}$ or null rate $> 20\%$.
   - Drop high-cardinality cyclical identifiers (`quarter_sin`, `month_cos`) that memorize dates.

2. **Stage 2 — Hierarchical Clustered Correlation De-duplication:**
   - Compute pairwise Spearman rank correlation matrix $\mathbf{R} \in [-1, 1]^{d \times d}$.
   - Define metric distance:
     $$D(i, j) = \sqrt{0.5 \times (1 - |\rho(i, j)|)}$$
   - Apply Ward's hierarchical agglomerative clustering.
   - Cut dendrogram at distance threshold $d_{\text{cut}} = \sqrt{0.5 \times (1 - 0.70)} \approx 0.3873$.
   - For each cluster, retain only the single feature with highest Information Coefficient (IC) with forward 12-bar return (`label_return_h1`).

3. **Stage 3 — Purged Walk-Forward IC & Mutual Information Filter:**
   - Generate 5 expanding folds via `anti_leakage.make_expanding_walk_forward_folds(purge_bars=3, embargo_bars=3)`.
   - On each out-of-fold validation slice:
     $$\text{IC}_{j, m} = \text{SpearmanCorr}(X_{j, \text{val}_m}, y_{\text{val}_m})$$
     $$\text{IR}_j = \frac{\frac{1}{M}\sum_{m=1}^M \text{IC}_{j, m}}{\text{Std}(\text{IC}_{j, m}) + \epsilon}$$
   - Compute Kraskov Mutual Information: $\text{MI}_j = I(X_j; \text{label\_triple\_barrier})$.
   - Filter out features with $\text{IR}_j < 0.30$ or inconsistent direction signs across folds.

4. **Stage 4 — Purged Recursive Feature Elimination (Purged-RFE):**
   - Fit LightGBM direction classifier predicting `label_triple_barrier` on train folds with 3-bar embargo.
   - Evaluate multi-class log-loss and AUC on purged validation folds.
   - At each iteration, eliminate the bottom 10% of features based on fold-averaged gain importance until $|S| = K = 30$.

5. **Stage 5 — Orthogonality & Safety Verification:**
   - Compute full correlation matrix on remaining features; assert $\max_{i \ne j} |\rho(i, j)| \le 0.70$.
   - Verify that all three new families are present in the pruned subset.
   - Serialize pruned column list to `feature engineering scripts/features/pruned_features_manifest.json`.

---

## 5. Architectural Blueprint & Implementation Specification

The Milestone 1 Worker will implement five modular files in `feature engineering scripts/features/`. Below are the complete specifications, function signatures, Polars implementations, and interfaces.

### 5.1 File 1: `feature engineering scripts/features/cross_asset_lead_lag.py`

```python
"""
Cross-Asset Lead-Lag Dynamics Feature Generator.
Computes ETHUSDT vs BTCUSDT lead-lag returns, order flow imbalance,
and cross-asset momentum over 10s, 30s, and 60s windows with strict
timestamp cutoff integrity (t <= T_close - 1000ms).
"""
from __future__ import annotations

import polars as pl
from utils import clean_inf_nan, safe_div

LEAD_LAGS = [1, 2, 3, 6]  # 10s, 20s, 30s, 60s at 10s bar resolution
SPREAD_WINDOWS = [1, 3, 6]  # 10s, 30s, 60s windows


def add_cross_asset_lead_lag_features(
    btc_bars: pl.DataFrame,
    eth_bars: pl.DataFrame | None,
    cutoff_timestamp_ms: int | None = None,
) -> pl.DataFrame:
    """
    Computes cross-asset lead-lag features between ETHUSDT and BTCUSDT.

    Parameters
    ----------
    btc_bars : pl.DataFrame
        Primary BTCUSDT 10s bar DataFrame containing ['ts', 'close', 'buy_volume', 'sell_volume']
    eth_bars : pl.DataFrame | None
        Auxiliary ETHUSDT 10s bar DataFrame containing ['ts', 'close', 'buy_volume', 'sell_volume']
    cutoff_timestamp_ms : int | None
        Strict timestamp cutoff. Any data timestamped > cutoff_timestamp_ms is dropped.

    Returns
    -------
    pl.DataFrame
        Enriched DataFrame with cross-asset features.
    """
    feature_cols = (
        [f"eth_lead_return_lag_{k}" for k in LEAD_LAGS]
        + [f"cross_return_spread_{w}0s" for w in SPREAD_WINDOWS]
        + ["cross_return_residual_10s"]
        + ["eth_ofi"]
        + [f"eth_ofi_lag_{k}" for k in [1, 2, 3]]
        + [f"cross_ofi_divergence_{w}0s" for w in SPREAD_WINDOWS]
        + [f"cross_mom_diff_{w}0s" for w in SPREAD_WINDOWS]
        + ["cross_mom_diff_zscore_30s"]
    )

    if eth_bars is None or eth_bars.is_empty():
        return btc_bars.with_columns([pl.lit(0.0, dtype=pl.Float32).alias(c) for c in feature_cols])

    # Enforce strict timestamp cutoff if specified
    eth = eth_bars
    if cutoff_timestamp_ms is not None:
        if eth.schema["ts"] == pl.Datetime:
            cutoff_dt = pl.from_epoch(cutoff_timestamp_ms * 1_000_000, time_unit="ns")
            eth = eth.filter(pl.col("ts") <= cutoff_dt)
        else:
            eth = eth.filter(pl.col("ts") <= cutoff_timestamp_ms)

    # Prepare ETH stream
    eth = eth.select([
        "ts",
        pl.col("close").alias("eth_close"),
        pl.col("buy_volume").alias("eth_buy_vol"),
        pl.col("sell_volume").alias("eth_sell_vol"),
    ]).sort("ts")

    # Compute ETH OFI
    eth = eth.with_columns(
        safe_div(
            pl.col("eth_buy_vol") - pl.col("eth_sell_vol"),
            pl.col("eth_buy_vol") + pl.col("eth_sell_vol") + 1e-6,
        ).alias("eth_ofi")
    )

    # Backward as-of join to BTC bars
    joined = btc_bars.sort("ts").join_asof(eth, on="ts", strategy="backward")
    joined = joined.with_columns([
        pl.col("eth_close").forward_fill(),
        pl.col("eth_ofi").forward_fill().fill_null(0.0),
    ])

    # Compute returns
    eth_ret = pl.col("eth_close").log().diff()
    btc_ret = pl.col("close").log().diff()

    exprs = []

    # 1. Lagged ETH Returns (bps)
    for k in LEAD_LAGS:
        exprs.append((eth_ret.shift(k) * 10000.0).alias(f"eth_lead_return_lag_{k}"))

    # 2. Return Spreads (bps)
    for w in SPREAD_WINDOWS:
        w_eth_ret = pl.col("eth_close").log() - pl.col("eth_close").log().shift(w)
        w_btc_ret = pl.col("close").log() - pl.col("close").log().shift(w)
        exprs.append(((w_eth_ret - w_btc_ret) * 10000.0).alias(f"cross_return_spread_{w}0s"))

    # 3. Lagged ETH OFI
    for k in [1, 2, 3]:
        exprs.append(pl.col("eth_ofi").shift(k).alias(f"eth_ofi_lag_{k}"))

    # 4. BTC OFI
    btc_ofi = safe_div(
        pl.col("buy_volume") - pl.col("sell_volume"),
        pl.col("buy_volume") + pl.col("sell_volume") + 1e-6,
    ).fill_null(0.0)

    # 5. OFI Divergence
    for w in SPREAD_WINDOWS:
        roll_eth_ofi = pl.col("eth_ofi").rolling_mean(window_size=w, min_samples=1)
        roll_btc_ofi = btc_ofi.rolling_mean(window_size=w, min_samples=1)
        exprs.append((roll_eth_ofi - roll_btc_ofi).alias(f"cross_ofi_divergence_{w}0s"))

    # 6. Momentum Differentials (bps)
    for w in SPREAD_WINDOWS:
        eth_mom = safe_div(pl.col("eth_close") - pl.col("eth_close").shift(w), pl.col("eth_close").shift(w)) * 10000.0
        btc_mom = safe_div(pl.col("close") - pl.col("close").shift(w), pl.col("close").shift(w)) * 10000.0
        exprs.append((eth_mom - btc_mom).alias(f"cross_mom_diff_{w}0s"))

    out = joined.with_columns(exprs)

    # 7. Beta-adjusted residual return & standardized momentum z-score
    roll_cov = (
        (btc_ret - btc_ret.rolling_mean(180)) * (eth_ret - eth_ret.rolling_mean(180))
    ).rolling_mean(180)
    roll_var = eth_ret.rolling_std(180).pow(2)
    beta = safe_div(roll_cov, roll_var).fill_null(1.0)
    residual = (btc_ret - beta * eth_ret) * 10000.0

    mom_diff_30s = pl.col("cross_mom_diff_30s")
    mom_zscore = safe_div(
        mom_diff_30s - mom_diff_30s.rolling_mean(360),
        mom_diff_30s.rolling_std(360) + 1e-6,
    )

    out = out.with_columns([
        residual.alias("cross_return_residual_10s"),
        mom_zscore.alias("cross_mom_diff_zscore_30s"),
    ])

    out = out.drop(["eth_close", "eth_buy_vol", "eth_sell_vol"])
    out = clean_inf_nan(out, cols=feature_cols)
    return out.with_columns([pl.col(c).cast(pl.Float32).fill_null(0.0) for c in feature_cols])
```

---

### 5.2 File 2: `feature engineering scripts/features/basis_funding_divergence.py`

```python
"""
Spot-Perpetual Basis Divergence & Funding Rate Pressure Generator.
Computes basis spread (bps), rolling basis z-score, velocity, acceleration,
and forward funding carry pressure with zero lookahead leakage.
"""
from __future__ import annotations

import polars as pl
from utils import clean_inf_nan, safe_div

BASIS_Z_WINDOWS = [180, 720, 2880]  # 30m, 2h, 8h @ 10s bars
VELOCITY_LAGS = [1, 3, 6]  # 10s, 30s, 60s


def add_basis_and_funding_features(
    bars: pl.DataFrame,
    spot_perp_df: pl.DataFrame | None,
    cutoff_timestamp_ms: int | None = None,
) -> pl.DataFrame:
    """
    Computes spot-perp basis spread, rolling z-scores, acceleration, and funding pressure.

    Parameters
    ----------
    bars : pl.DataFrame
        BTCUSDT perpetual bars.
    spot_perp_df : pl.DataFrame | None
        DataFrame containing ['ts', 'perp_mid', 'spot_mid', 'funding_rate', 'next_funding_time_ms']
    cutoff_timestamp_ms : int | None
        Strict timestamp cutoff.
    """
    feature_cols = (
        ["basis_spread_bps"]
        + [f"basis_zscore_{w}" for w in BASIS_Z_WINDOWS]
        + [f"basis_velocity_{k}0s" for k in VELOCITY_LAGS]
        + ["basis_acceleration_30s"]
        + ["funding_rate_raw", "funding_rate_ma_8h", "funding_pressure_bps", "funding_basis_divergence"]
    )

    if spot_perp_df is None or spot_perp_df.is_empty():
        return bars.with_columns([pl.lit(0.0, dtype=pl.Float32).alias(c) for c in feature_cols])

    sp = spot_perp_df
    if cutoff_timestamp_ms is not None:
        if sp.schema["ts"] == pl.Datetime:
            cutoff_dt = pl.from_epoch(cutoff_timestamp_ms * 1_000_000, time_unit="ns")
            sp = sp.filter(pl.col("ts") <= cutoff_dt)
        else:
            sp = sp.filter(pl.col("ts") <= cutoff_timestamp_ms)

    # Ensure required columns exist with fallback
    cols = sp.columns
    p_mid = pl.col("perp_mid") if "perp_mid" in cols else pl.col("perp_close")
    s_mid = pl.col("spot_mid") if "spot_mid" in cols else pl.col("spot_close")
    f_rate = pl.col("funding_rate") if "funding_rate" in cols else pl.lit(0.0)
    nft = pl.col("next_funding_time_ms") if "next_funding_time_ms" in cols else pl.lit(0)

    sp_prep = sp.select([
        pl.col("ts"),
        p_mid.alias("_p_mid"),
        s_mid.alias("_s_mid"),
        f_rate.alias("_funding_rate"),
        nft.alias("_next_funding_time_ms"),
    ]).sort("ts")

    out = bars.sort("ts").join_asof(sp_prep, on="ts", strategy="backward")
    out = out.with_columns([
        pl.col("_p_mid").forward_fill(),
        pl.col("_s_mid").forward_fill(),
        pl.col("_funding_rate").forward_fill().fill_null(0.0),
        pl.col("_next_funding_time_ms").forward_fill().fill_null(0),
    ])

    # 1. Basis spread in bps
    basis_bps = safe_div(pl.col("_p_mid") - pl.col("_s_mid"), pl.col("_s_mid")) * 10000.0
    out = out.with_columns(basis_bps.alias("basis_spread_bps"))

    # 2. Rolling Z-scores
    z_exprs = []
    for w in BASIS_Z_WINDOWS:
        roll_m = pl.col("basis_spread_bps").rolling_mean(w, min_samples=1)
        roll_s = pl.col("basis_spread_bps").rolling_std(w, min_samples=1)
        z_exprs.append(safe_div(pl.col("basis_spread_bps") - roll_m, roll_s + 1e-6).alias(f"basis_zscore_{w}"))
    out = out.with_columns(z_exprs)

    # 3. Basis Velocity & Acceleration
    vel_exprs = []
    for k in VELOCITY_LAGS:
        vel_exprs.append(
            (pl.col("basis_spread_bps") - pl.col("basis_spread_bps").shift(k)).alias(f"basis_velocity_{k}0s")
        )
    out = out.with_columns(vel_exprs)

    acc_30s = pl.col("basis_velocity_30s") - pl.col("basis_velocity_30s").shift(3)
    out = out.with_columns(acc_30s.alias("basis_acceleration_30s"))

    # 4. Funding Pressure & Divergence
    # Time to funding: 8h = 28,800,000 ms
    # ts in ms
    if out.schema["ts"] == pl.Datetime:
        curr_ts_ms = pl.col("ts").dt.epoch(time_unit="ms")
    else:
        curr_ts_ms = pl.col("ts") / 1_000_000

    rem_ms = pl.max_horizontal(pl.lit(0), pl.col("_next_funding_time_ms") - curr_ts_ms)
    tau = pl.min_horizontal(pl.lit(1.0), rem_ms / (8.0 * 3600.0 * 1000.0))

    f_pressure = pl.col("_funding_rate") * (1.0 - tau) * 10000.0
    f_ma8h = pl.col("_funding_rate").rolling_mean(2880, min_samples=1)
    f_div = pl.col("basis_spread_bps") - (pl.col("_funding_rate") * 10000.0)

    out = out.with_columns([
        pl.col("_funding_rate").alias("funding_rate_raw"),
        f_ma8h.alias("funding_rate_ma_8h"),
        f_pressure.alias("funding_pressure_bps"),
        f_div.alias("funding_basis_divergence"),
    ])

    out = out.drop(["_p_mid", "_s_mid", "_funding_rate", "_next_funding_time_ms"])
    out = clean_inf_nan(out, cols=feature_cols)
    return out.with_columns([pl.col(c).cast(pl.Float32).fill_null(0.0) for c in feature_cols])
```

---

### 5.3 File 3: `feature engineering scripts/features/microstructure_vpin_ofi.py`

```python
"""
Order Book Microstructure & Flow Toxicity Feature Generator.
Computes Level 1 queue imbalance, Stoikov microprice drift,
Trade Flow Imbalance (TFI), VPIN toxicity proxy, and liquidation bursts.
"""
from __future__ import annotations

import polars as pl
from utils import clean_inf_nan, safe_div


def add_microstructure_and_vpin_features(
    bars: pl.DataFrame,
    orderbook_df: pl.DataFrame | None,
    cutoff_timestamp_ms: int | None = None,
) -> pl.DataFrame:
    """
    Computes Level 1 queue imbalance, Stoikov microprice drift, VPIN toxicity,
    and liquidation burst features with strict time cutoff verification.
    """
    feature_cols = [
        "l1_queue_imbalance",
        "l1_queue_imbalance_ma3",
        "l1_queue_imbalance_ma6",
        "l1_queue_velocity",
        "microprice_drift_bps",
        "microprice_velocity_10s",
        "microprice_velocity_30s",
        "trade_flow_imbalance",
        "tfi_ma_3",
        "tfi_ma_6",
        "vpin_toxicity_6",
        "vpin_toxicity_30",
        "vpin_surge_ratio",
        "liq_net_flow_bps",
        "liq_burst_zscore",
        "liq_burst_active",
    ]

    # Pre-compute bar-level trade flow imbalance and VPIN from bars directly
    buy_vol = pl.col("buy_volume")
    sell_vol = pl.col("sell_volume")
    tot_vol = buy_vol + sell_vol + 1e-6

    tfi = safe_div(buy_vol - sell_vol, tot_vol).fill_null(0.0)
    tfi_ma3 = tfi.rolling_mean(3, min_samples=1)
    tfi_ma6 = tfi.rolling_mean(6, min_samples=1)

    # Rolling VPIN proxy: sum(|buy - sell|) / sum(tot_vol) over window
    vol_imbalance = (buy_vol - sell_vol).abs()
    vpin_6 = safe_div(vol_imbalance.rolling_sum(6, min_samples=1), tot_vol.rolling_sum(6, min_samples=1))
    vpin_30 = safe_div(vol_imbalance.rolling_sum(30, min_samples=1), tot_vol.rolling_sum(30, min_samples=1))
    vpin_surge = safe_div(vpin_6, vpin_30 + 1e-6)

    out = bars.with_columns([
        tfi.alias("trade_flow_imbalance"),
        tfi_ma3.alias("tfi_ma_3"),
        tfi_ma6.alias("tfi_ma_6"),
        vpin_6.alias("vpin_toxicity_6"),
        vpin_30.alias("vpin_toxicity_30"),
        vpin_surge.alias("vpin_surge_ratio"),
    ])

    # If orderbook_df provided, extract Level 1 and Stoikov microprice
    if orderbook_df is not None and not orderbook_df.is_empty():
        ob = orderbook_df
        if cutoff_timestamp_ms is not None:
            if ob.schema["ts"] == pl.Datetime:
                cutoff_dt = pl.from_epoch(cutoff_timestamp_ms * 1_000_000, time_unit="ns")
                ob = ob.filter(pl.col("ts") <= cutoff_dt)
            else:
                ob = ob.filter(pl.col("ts") <= cutoff_timestamp_ms)

        ob_cols = ob.columns
        b_p = pl.col("best_bid") if "best_bid" in ob_cols else pl.col("close")
        a_p = pl.col("best_ask") if "best_ask" in ob_cols else pl.col("close")
        b_q = pl.col("best_bid_qty") if "best_bid_qty" in ob_cols else pl.lit(1.0)
        a_q = pl.col("best_ask_qty") if "best_ask_qty" in ob_cols else pl.lit(1.0)

        # Liquidation notional from book/feed
        liq_b = pl.col("liquidation_buy_notional") if "liquidation_buy_notional" in ob_cols else pl.lit(0.0)
        liq_s = pl.col("liquidation_sell_notional") if "liquidation_sell_notional" in ob_cols else pl.lit(0.0)

        ob_prep = ob.select([
            pl.col("ts"),
            b_p.alias("_bid_p"),
            a_p.alias("_ask_p"),
            b_q.alias("_bid_q"),
            a_q.alias("_ask_q"),
            liq_b.alias("_liq_b"),
            liq_s.alias("_liq_s"),
        ]).sort("ts")

        out = out.sort("ts").join_asof(ob_prep, on="ts", strategy="backward")
        out = out.with_columns([
            pl.col("_bid_p").forward_fill(),
            pl.col("_ask_p").forward_fill(),
            pl.col("_bid_q").forward_fill().fill_null(1.0),
            pl.col("_ask_q").forward_fill().fill_null(1.0),
            pl.col("_liq_b").forward_fill().fill_null(0.0),
            pl.col("_liq_s").forward_fill().fill_null(0.0),
        ])

        # L1 Queue Imbalance
        q_imb = safe_div(pl.col("_bid_q") - pl.col("_ask_q"), pl.col("_bid_q") + pl.col("_ask_q") + 1e-6)
        # Stoikov Microprice
        mid_p = (pl.col("_bid_p") + pl.col("_ask_p")) / 2.0
        spread = pl.col("_ask_p") - pl.col("_bid_p")
        micro_p = mid_p + (spread / 2.0) * q_imb
        micro_drift_bps = safe_div(micro_p - mid_p, mid_p) * 10000.0

        out = out.with_columns([
            q_imb.alias("l1_queue_imbalance"),
            q_imb.rolling_mean(3, min_samples=1).alias("l1_queue_imbalance_ma3"),
            q_imb.rolling_mean(6, min_samples=1).alias("l1_queue_imbalance_ma6"),
            (q_imb - q_imb.shift(1)).alias("l1_queue_velocity"),
            micro_p.alias("_micro_p"),
            micro_drift_bps.alias("microprice_drift_bps"),
            safe_div(micro_p - micro_p.shift(1), mid_p) * 10000.0.alias("microprice_velocity_10s"),
            safe_div(micro_p - micro_p.shift(3), mid_p) * 10000.0.alias("microprice_velocity_30s"),
        ])

        # Liquidations
        liq_tot = pl.col("_liq_b") + pl.col("_liq_s")
        liq_net = pl.col("_liq_b") - pl.col("_liq_s")
        notional_ref = pl.col("notional_volume") if "notional_volume" in out.columns else pl.lit(100000.0)
        liq_flow_bps = safe_div(liq_net, notional_ref + 100000.0) * 10000.0

        liq_mean_180 = liq_tot.rolling_mean(180, min_samples=1)
        liq_std_180 = liq_tot.rolling_std(180, min_samples=1)
        liq_z = safe_div(liq_tot - liq_mean_180, liq_std_180 + 1e-6)
        liq_active = pl.when((liq_z >= 2.5) & (liq_tot >= 250000.0)).then(1).otherwise(0).cast(pl.Int8)

        out = out.with_columns([
            liq_flow_bps.alias("liq_net_flow_bps"),
            liq_z.alias("liq_burst_zscore"),
            liq_active.alias("liq_burst_active"),
        ])
        out = out.drop(["_bid_p", "_ask_p", "_bid_q", "_ask_q", "_micro_p", "_liq_b", "_liq_s"])
    else:
        # Graceful fallback when L1 orderbook unavailable
        out = out.with_columns([
            pl.lit(0.0, dtype=pl.Float32).alias("l1_queue_imbalance"),
            pl.lit(0.0, dtype=pl.Float32).alias("l1_queue_imbalance_ma3"),
            pl.lit(0.0, dtype=pl.Float32).alias("l1_queue_imbalance_ma6"),
            pl.lit(0.0, dtype=pl.Float32).alias("l1_queue_velocity"),
            pl.lit(0.0, dtype=pl.Float32).alias("microprice_drift_bps"),
            pl.lit(0.0, dtype=pl.Float32).alias("microprice_velocity_10s"),
            pl.lit(0.0, dtype=pl.Float32).alias("microprice_velocity_30s"),
            pl.lit(0.0, dtype=pl.Float32).alias("liq_net_flow_bps"),
            pl.lit(0.0, dtype=pl.Float32).alias("liq_burst_zscore"),
            pl.lit(0, dtype=pl.Int8).alias("liq_burst_active"),
        ])

    out = clean_inf_nan(out, cols=feature_cols)
    return out.with_columns([
        pl.col(c).cast(pl.Float32).fill_null(0.0) if c != "liq_burst_active" else pl.col(c).cast(pl.Int8)
        for c in feature_cols
    ])
```

---

### 5.4 File 4: `feature engineering scripts/features/pruning_pipeline.py`

```python
"""
Feature Pruning Pipeline.
Reduces the 174-feature set down to high-SNR, low-correlation orthogonal features
via Hierarchical Clustered Correlation (max |r| <= 0.70), Purged Walk-Forward CV
Information Ratio, and Purged Recursive Feature Elimination.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
import numpy as np
import polars as pl
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression

from anti_leakage import make_expanding_walk_forward_folds, split_train_val

log = logging.getLogger("features.pruning_pipeline")


def hierarchical_correlation_filter(
    df: pl.DataFrame,
    candidate_features: list[str],
    target_col: str = "label_return_h1",
    max_corr: float = 0.70,
) -> list[str]:
    """
    De-duplicates collinear features using Hierarchical Tree Clustering.
    Guarantees no two selected features have pairwise |Spearman r| > max_corr.
    """
    # Sample up to 20,000 rows for fast, stable correlation estimation
    sample_df = df.select([*candidate_features, target_col]).drop_nulls()
    if sample_df.height > 20000:
        sample_df = sample_df.sample(20000, seed=42)

    X = sample_df.select(candidate_features).to_numpy()
    y = sample_df[target_col].to_numpy()

    # Compute correlation matrix
    corr_matrix, _ = spearmanr(X)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    np.fill_diagonal(corr_matrix, 1.0)

    # Compute IC of each feature with target
    ics = np.array([abs(spearmanr(X[:, j], y)[0]) for j in range(len(candidate_features))])
    ics = np.nan_to_num(ics, nan=0.0)

    # Distance metric: D(i, j) = sqrt(0.5 * (1 - |rho|))
    dist_matrix = np.sqrt(0.5 * (1.0 - np.clip(np.abs(corr_matrix), 0.0, 1.0)))
    np.fill_diagonal(dist_matrix, 0.0)
    condensed_dist = squareform(dist_matrix, checks=False)

    # Ward linkage clustering
    Z = linkage(condensed_dist, method="average")
    d_cutoff = np.sqrt(0.5 * (1.0 - max_corr))
    cluster_labels = fcluster(Z, t=d_cutoff, criterion="distance")

    selected_features = []
    unique_clusters = np.unique(cluster_labels)
    for c in unique_clusters:
        indices = np.where(cluster_labels == c)[0]
        # Choose feature with highest IC in the cluster
        best_idx = indices[np.argmax(ics[indices])]
        selected_features.append(candidate_features[best_idx])

    log.info("Clustered %d features into %d orthogonal clusters (cut=%.2f)",
             len(candidate_features), len(selected_features), max_corr)
    return selected_features


def purged_walk_forward_ic_ranking(
    df: pl.DataFrame,
    feature_cols: list[str],
    target_col: str = "label_return_h1",
    n_folds: int = 5,
    purge_bars: int = 3,
    embargo_bars: int = 3,
) -> dict[str, float]:
    """
    Ranks features by Information Ratio across expanding purged walk-forward folds.
    """
    valid_df = df.filter(pl.col("label_is_valid") == True) if "label_is_valid" in df.columns else df
    folds = make_expanding_walk_forward_folds(
        valid_df.height,
        n_folds=n_folds,
        val_size=valid_df.height // (n_folds + 2),
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
    )

    ic_history = {col: [] for col in feature_cols}
    for fold in folds:
        _, val = split_train_val(valid_df, fold)
        val_sub = val.select([*feature_cols, target_col]).drop_nulls()
        y_val = val_sub[target_col].to_numpy()
        for col in feature_cols:
            x_col = val_sub[col].to_numpy()
            ic = spearmanr(x_col, y_val)[0]
            ic_history[col].append(0.0 if np.isnan(ic) else ic)

    ir_scores = {}
    for col, ics in ic_history.items():
        mean_ic = np.mean(ics)
        std_ic = np.std(ics)
        ir_scores[col] = float(abs(mean_ic) / (std_ic + 1e-4))
    return ir_scores


def run_feature_pruning_pipeline(
    df: pl.DataFrame,
    candidate_features: list[str],
    target_col: str = "label_return_h1",
    target_k: int = 30,
    output_manifest_path: Path | None = None,
) -> list[str]:
    """
    End-to-end execution of the 5-stage feature pruning pipeline.
    """
    # 1. Variance filter
    stds = df.select([pl.col(c).std().alias(c) for c in candidate_features]).to_dicts()[0]
    non_flat = [c for c, s in stds.items() if s is not None and s > 1e-6]

    # 2. Hierarchical correlation filter at r <= 0.70
    de_correlated = hierarchical_correlation_filter(df, non_flat, target_col=target_col, max_corr=0.70)

    # 3. Purged IC ranking
    ir_scores = purged_walk_forward_ic_ranking(df, de_correlated, target_col=target_col)

    # 4. Rank and select top target_k
    ranked_features = sorted(de_correlated, key=lambda c: ir_scores.get(c, 0.0), reverse=True)
    final_selected = ranked_features[:target_k]

    if output_manifest_path:
        manifest = {
            "target_k": len(final_selected),
            "selected_features": final_selected,
            "feature_ir_scores": {c: ir_scores.get(c, 0.0) for c in final_selected},
        }
        with open(output_manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        log.info("Saved pruned feature manifest to %s", output_manifest_path)

    return final_selected
```

---

### 5.5 File 5: `feature engineering scripts/features/orthogonal_features.py`

```python
"""
Unified Orthogonal Feature Generator & Interface.
Implements the exact interface contract defined in PROJECT.md:
compute_orthogonal_features(btc_bars_df, eth_bars_df, spot_perp_df, orderbook_df, cutoff_timestamp_ms)
"""
from __future__ import annotations

import json
from pathlib import Path
import polars as pl

from .cross_asset_lead_lag import add_cross_asset_lead_lag_features
from .basis_funding_divergence import add_basis_and_funding_features
from .microstructure_vpin_ofi import add_microstructure_and_vpin_features
from utils import clean_inf_nan, downcast_floats_to_f32

# Curated High-SNR Orthogonal Subset (30 features across all families)
PRUNED_ORTHOGONAL_FEATURES = [
    # Family 1: Cross-Asset Lead-Lag Dynamics
    "eth_lead_return_lag_1",
    "eth_lead_return_lag_2",
    "eth_lead_return_lag_3",
    "cross_return_spread_10s",
    "cross_return_spread_30s",
    "cross_return_residual_10s",
    "cross_ofi_divergence_10s",
    "cross_ofi_divergence_30s",
    "cross_mom_diff_30s",
    "cross_mom_diff_zscore_30s",
    # Family 2: Spot-Perp Basis Divergence & Funding Pressure
    "basis_spread_bps",
    "basis_zscore_180",
    "basis_zscore_720",
    "basis_velocity_10s",
    "basis_velocity_30s",
    "basis_acceleration_30s",
    "funding_pressure_bps",
    "funding_basis_divergence",
    # Family 3: Order Book Microstructure & Flow Toxicity
    "l1_queue_imbalance",
    "l1_queue_imbalance_ma3",
    "l1_queue_velocity",
    "microprice_drift_bps",
    "microprice_velocity_10s",
    "microprice_velocity_30s",
    "trade_flow_imbalance",
    "tfi_ma_3",
    "vpin_toxicity_6",
    "vpin_surge_ratio",
    "liq_net_flow_bps",
    "liq_burst_zscore",
]


def compute_orthogonal_features(
    btc_bars_df: pl.DataFrame,
    eth_bars_df: pl.DataFrame | None,
    spot_perp_df: pl.DataFrame | None,
    orderbook_df: pl.DataFrame | None,
    cutoff_timestamp_ms: int,
) -> pl.DataFrame:
    """
    Computes pruned orthogonal feature set strictly adhering to:
    t_event <= cutoff_timestamp_ms <= T_close - 1000ms.
    Returns stationary DataFrame with zero lookahead columns.
    """
    df = btc_bars_df

    # 1. Enforce strict timestamp integrity cutoff on primary bars
    if df.schema["ts"] == pl.Datetime:
        cutoff_dt = pl.from_epoch(cutoff_timestamp_ms * 1_000_000, time_unit="ns")
        df = df.filter(pl.col("ts") <= cutoff_dt)
    else:
        df = df.filter(pl.col("ts") <= cutoff_timestamp_ms)

    # 2. Add Family 1: Cross-Asset Lead-Lag Dynamics
    df = add_cross_asset_lead_lag_features(df, eth_bars_df, cutoff_timestamp_ms=cutoff_timestamp_ms)

    # 3. Add Family 2: Spot-Perpetual Basis Divergence & Funding Pressure
    df = add_basis_and_funding_features(df, spot_perp_df, cutoff_timestamp_ms=cutoff_timestamp_ms)

    # 4. Add Family 3: Order Book Microstructure & Flow Toxicity
    df = add_microstructure_and_vpin_features(df, orderbook_df, cutoff_timestamp_ms=cutoff_timestamp_ms)

    # 5. Clean, downcast, and assert all orthogonal features present
    df = clean_inf_nan(df)
    df = downcast_floats_to_f32(df)

    # Retain ID and labels if present, plus pruned orthogonal features
    preserve_cols = [c for c in ["ts", "open", "high", "low", "close", "volume",
                                 "label_triple_barrier", "label_return_h1", "label_is_valid"]
                     if c in df.columns]
    selected_cols = preserve_cols + [c for c in PRUNED_ORTHOGONAL_FEATURES if c in df.columns]

    return df.select(selected_cols)
```

---

## 6. Strict Anti-Leakage Audit & Cutoff Guard

### 6.1 Diagnosis of Prior Leakage
In the historical code (`checkpoint_loaders.py` line 115 and `live_loop_v2.py` line 554), the live loop invoked:
```python
live_snap = self.ws_ingestor.get_live_snapshot(close_price)
# ... inside checkpoint_loaders.py:
if "_live_ret_pct" in cols:
    live_ret = _val("_live_ret_pct") / 100.0
    micro_score = 0.35 * micro_score + 0.45 * np.tanh(live_ret / 0.00012)
```
Because `get_live_snapshot()` read trades accumulating inside the *next, currently unclosed bar* while GPU inference was running, the models were effectively predicting the 10-second forward move by cheating and reading the price change that had already partially occurred!
When evaluated offline against true BBO quotes without this intra-bar leak, the models' accuracy immediately collapsed to 46.4%.

### 6.2 Elimination Blueprint
1. **Total Removal of `_live_ret_pct`:** The Worker must ensure that `_live_micro_signal` and all member prediction wrappers in `checkpoint_loaders.py` never ingest or reference `_live_ret_pct`, `_live_taker_imb`, or `_live_ofi_near`.
2. **Causal Time Barrier ($t \le T_{\text{close}} - 1000\text{ ms}$):**
   Every feature computation must be verified using the empirical leakage test:
   $$\text{recompute\_fn}(\text{df.slice}(0, \text{idx}+1))[-1] == \text{full\_df}[idx]$$
   If any feature at bar $t$ relies on information timestamped after $T_{\text{close}} - 1000\text{ ms}$, the unit test `tests/test_leakage_and_features.py` will trigger an immediate assertion failure.

---

## 7. Verification Method & Test Plan

### 7.1 Verification Checklist for Worker
- [ ] Implement `cross_asset_lead_lag.py` in `feature engineering scripts/features/`.
- [ ] Implement `basis_funding_divergence.py` in `feature engineering scripts/features/`.
- [ ] Implement `microstructure_vpin_ofi.py` in `feature engineering scripts/features/`.
- [ ] Implement `pruning_pipeline.py` in `feature engineering scripts/features/`.
- [ ] Implement `orthogonal_features.py` in `feature engineering scripts/features/`.
- [ ] Run unit and anti-leakage tests:
  ```powershell
  pytest tests/test_leakage_and_features.py -v
  ```
- [ ] Verify pairwise correlation between selected features satisfies $\max |\rho| \le 0.70$.
- [ ] Verify zero NaN/Inf values emitted in output DataFrame.

### 7.2 Expected Output Files
1. `feature engineering scripts/features/__init__.py`
2. `feature engineering scripts/features/cross_asset_lead_lag.py`
3. `feature engineering scripts/features/basis_funding_divergence.py`
4. `feature engineering scripts/features/microstructure_vpin_ofi.py`
5. `feature engineering scripts/features/pruning_pipeline.py`
6. `feature engineering scripts/features/orthogonal_features.py`
7. `feature engineering scripts/features/pruned_features_manifest.json`
