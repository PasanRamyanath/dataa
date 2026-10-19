# Comprehensive Specification & Empirical Ledger Mining Report
**Project:** BTCUSDT 10-Second Alpha & Execution Overhaul  
**Miner:** Spec & Ledger Miner (`teamwork_preview_spec_miner`)  
**Target Path:** `D:\backup\Binance data\.agents\teamwork\spec_miner_survey\survey_spec_report.md`  
**Date:** 2026-10-18  

---

## 1. Executive Summary & Authoritative Sources

This report provides an exhaustive, mathematically rigorous specification and empirical ledger mining audit for the overhaul of the BTCUSDT 10-second-bar futures prediction and execution pipeline in `D:\backup\Binance data`.

### Authoritative Specification Sources Mined:
1. **`ORIGINAL_REQUEST.md`** (`D:\backup\Binance data\.agents\teamwork\ORIGINAL_REQUEST.md`): Authoritative project mandate, scope requirements R1–R4, and acceptance criteria AC1–AC5.
2. **`scratch/live_45m_eval_ledger.jsonl`**: The primary 270-bar (45-minute) live telemetry event ledger recorded from Binance USD(S)-M Futures (`BTCUSDT`) on 2026-10-17 17:54:40 to 18:42:41 UTC.
3. **`result.md`** & **`feature engineering scripts/scratch/rigorous_evaluation_audit.py`**: The quantitative evaluation audit suite, formalizing BBO mid-price benchmarking, Wilson score confidence intervals, exact binomial hypothesis testing, run clustering, drift-adjusted alpha, Brier calibration, and transaction cost modeling.
4. **`scratch/live_30m_eval_summary.json`** & **`scratch/live_30m_eval_ledger.jsonl`**: The secondary 193-bar live telemetry session, recording ultra-sniper chop vetoes, 5 closed trades, and member model hit rates.
5. **`feature engineering scripts/inference/checkpoint_loaders.py`**: Model inference wrappers and checkpoint deserialization logic exposing the intra-bar microprice leakage (`_live_micro_signal` / `_live_ret_pct`) and shared ensemble feature contamination.
6. **`feature engineering scripts/streaming/live_loop_v2.py`**: Execution loop exposing the hard piecewise clamping and bimodal distribution collapse of the meta-labeling filter.
7. **`feature engineering scripts/labeling.py`** & **`feature engineering scripts/config.py`**: Triple-barrier labeling definitions revealing horizon discrepancies between training labels (360 bars = 1 hr), previous execution (18 bars = 180s), and actual empirical peak alpha (3 bars = 30s).
8. **`feature engineering scripts/common/feature_contract.py`** & **`websocket-architecture.md`**: Schema contract defining the 226 total columns, 174 active stationary features, dead features, and WebSocket streaming constraints.

---

## 2. Deep Empirical Audit of Live Telemetry Ledgers

### 2.1 The 45-Minute Live Ledger (`scratch/live_45m_eval_ledger.jsonl`)
- **Total Duration:** 2,880.46 seconds (48.01 minutes total monitoring session).
- **Total Bars Recorded:** 270 ten-second bars ($t_1$ to $t_{270}$).
- **Exchange & Instrument:** Binance USD(S)-M Futures, `BTCUSDT` Perpetual.
- **Reference Price:** Real-time Binance BBO (Best Bid and Offer) Mid-Price:
  $$P_{\text{BBO\_mid}} = \frac{P_{\text{best\_bid}} + P_{\text{best\_ask}}}{2}$$
- **Market Regime Observed:** Secular upward drift. BTC BBO mid moved from **$\$81,329.05$** to **$\$81,533.85$** ($+204.80\text{ USD}$, **$+25.18\text{ bps}$** total market drift, $+0.0933\text{ bps/bar}$). Exactly **$66.3\%$ of rolling 180-second windows ended UP**.
- **Distribution of Model Decisions:**
  - **FLAT (Class 1):** 242 bars ($89.63\%$ of all bars).
  - **Directional Calls (Classes 0 or 2):** 28 bars ($10.37\%$ of all bars).
    - **SHORT Calls (Class 0):** 21 bars ($75.0\%$ of directional calls).
    - **LONG Calls (Class 2):** 7 bars ($25.0\%$ of directional calls).
    - *Structural Bias:* Strong short bias (3:1 short vs long) in a rising market.

### 2.2 Re-Evaluation Against BBO Mid-Price vs Last-Trade Close
The previous headline accuracy claims of $60.7\% - 64.3\%$ were artifactually boosted by evaluating against `close_price` (last executed trade) rather than true BBO mid-price:

| Horizon | Lookahead | Directional Bars ($N$) | Ties ($\Delta = 0.00$) | Tie % | Raw Mid Hit Rate | Wilson 95% CI (Raw) | Raw Binomial p-val | Non-Tie Mid Hit Rate | Wilson 95% CI (Non-Tie) | Non-Tie p-val | Latency-Adj Hit Rate ($t+1\text{s}$) | Latency 95% CI | Latency p-val | Mean Raw Edge (bps) | Mean Latency Edge (bps) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **H1** | 10s (1 bar) | 28 | 12 | 42.86% | **46.43%** (13/28) | [29.5%, 64.2%] | 0.851 | **81.25%** (13/16) | [57.0%, 93.4%] | 0.021* | **44.44%** (8/18) | [24.6%, 66.3%] | 0.815 | $+0.245$ | $-0.063$ |
| **H2** | 20s (2 bars) | 28 | 6 | 21.43% | **53.57%** (15/28) | [35.8%, 70.5%] | 0.851 | **68.18%** (15/22) | [47.3%, 83.6%] | 0.134 | **54.55%** (12/22) | [34.7%, 73.1%] | 0.832 | $+0.236$ | $+0.041$ |
| **H3** | **30s (3 bars)** | **28** | **6** | **21.43%** | **46.43%** (13/28) | **[29.5%, 64.2%]** | **0.851** | **59.09%** (13/22) | **[38.7%, 76.7%]** | **0.523** | **52.94%** (9/17) | **[31.0%, 73.8%]** | **1.000** | **$+0.227$** | **$-0.176$** |
| **H6** | 60s (6 bars) | 28 | 4 | 14.29% | **39.29%** (11/28) | [23.6%, 57.6%] | 0.345 | **45.83%** (11/24) | [27.9%, 64.9%] | 0.839 | **37.50%** (9/24) | [21.2%, 57.3%] | 0.307 | $-0.125$ | $-0.388$ |
| **H12** | 120s (12 bars)| 28 | 3 | 10.71% | **32.14%** (9/28) | [17.9%, 50.7%] | 0.087 | **36.00%** (9/25) | [20.2%, 55.8%] | 0.230 | **32.00%** (8/25) | [17.2%, 51.6%] | 0.108 | $-0.562$ | $-0.712$ |
| **H18** | 180s (18 bars)| 28 | 1 | 3.57% | **32.14%** (9/28) | [17.9%, 50.7%] | 0.087 | **33.33%** (9/27) | [18.6%, 52.2%] | 0.122 | **29.63%** (8/27) | [15.3%, 49.5%] | 0.052 | $+0.034$ | $-0.215$ |

*Empirical Takeaway:* At the pre-registered 30-second primary horizon ($H3$), raw accuracy is $46.43\%$ ($p = 0.851$), non-tie accuracy is $59.09\%$ ($p = 0.523$), and latency-adjusted accuracy is $52.94\%$ ($p = 1.000$). None of these reject the null hypothesis of a fair coin toss ($p > 0.05$).

### 2.3 The 5 Authorized Trades Audit
Under the legacy `live_loop_v2.py` logic, an order was authorized when `act_probability >= 0.70`. Exactly 5 bars met this criterion: bars 65, 161, 162, 191, and 234. All 5 were SHORT calls:

| Bar Index | Signal | Entry Mid | Conviction $\tau$ | Gross 30s Move | Net 30s VIP0 Maker (-4 bps) | Net 30s Taker (-10 bps) | Gross 180s Move | Net 180s VIP0 Maker (-4 bps) | Net 180s Taker (-10 bps) | Exit Cause at 180s |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Bar 65** | SHORT | $\$81,455.65$ | 0.942 | $-1.44\text{ bps}$ | **$-5.44\text{ bps}$** | **$-11.44\text{ bps}$** | $-0.16\text{ bps}$ | **$-4.16\text{ bps}$** | **$-10.16\text{ bps}$** | Max Hold Expiry |
| **Bar 161** | SHORT | $\$81,496.35$ | 0.877 | $+1.18\text{ bps}$ | **$-2.82\text{ bps}$** | **$-8.82\text{ bps}$** | $-0.69\text{ bps}$ | **$-4.69\text{ bps}$** | **$-10.69\text{ bps}$** | Max Hold Expiry |
| **Bar 162** | SHORT | $\$81,486.85$ | 0.850 | $+0.74\text{ bps}$ | **$-3.26\text{ bps}$** | **$-9.26\text{ bps}$** | $-1.69\text{ bps}$ | **$-5.69\text{ bps}$** | **$-11.69\text{ bps}$** | Max Hold Expiry |
| **Bar 191** | SHORT | $\$81,518.85$ | 0.855 | $0.00\text{ bps}$ | **$-4.00\text{ bps}$** | **$-10.00\text{ bps}$** | $0.00\text{ bps}$ | **$-4.00\text{ bps}$** | **$-10.00\text{ bps}$** | Max Hold Expiry |
| **Bar 234** | SHORT | $\$81,518.85$ | 0.896 | $+0.44\text{ bps}$ | **$-3.56\text{ bps}$** | **$-9.56\text{ bps}$** | $-0.90\text{ bps}$ | **$-4.90\text{ bps}$** | **$-10.90\text{ bps}$** | Max Hold Expiry |

*Critical Diagnostic Takeaways:*
1. **100% Time-Stop Failures:** At 180 seconds, 100% of trades expired via max hold time stop because the fixed $+16.0\text{ bps}$ TP barrier was completely unreachable in normal 10s volatility.
2. **Gross 180s Drag:** All 5 trades lost gross at 180s because they shorted into a $+25.2\text{ bps}$ macro upward trend.
3. **Gross 30s Edge:** At the 30s horizon, 3 of 5 were profitable gross ($+1.18, +0.74, +0.44\text{ bps}$), 1 tie, 1 loss.
4. **Fee Annihilation:** Under taker fees (10 bps RT), 100% of trades lost heavily ($-8.82$ to $-11.69\text{ bps}$). Under Binance USD-M VIP0 maker fees (4 bps RT), all 5 trades still lost net ($-2.82$ to $-5.44\text{ bps}$) because gross moves were $\le 1.18\text{ bps}$, below the 4 bps cost hurdle.

---

## 3. Mathematical Formulations & Statistical Definitions

### 3.1 Directional Return & Accuracy Metrics
Let $P_{\text{mid}, t}$ denote the BBO mid-price at bar index $t$. The forward return at horizon $h$ (in basis points) is:
$$R_t(h) = \left( \frac{P_{\text{mid}, t+h} - P_{\text{mid}, t}}{P_{\text{mid}, t}} \right) \times 10,000$$

For a directional call $\hat{y}_t \in \{0, 2\}$ where $0 = \text{SHORT}$ and $2 = \text{LONG}$, the directional edge is:
$$\text{Edge}_t(h) = \begin{cases} +R_t(h), & \text{if } \hat{y}_t = 2 \text{ (LONG)} \\ -R_t(h), & \text{if } \hat{y}_t = 0 \text{ (SHORT)} \end{cases}$$

Microstructure ties occur when the mid-price does not move:
$$\text{Tie}_t(h) = \mathbb{I}(|\text{Edge}_t(h)| < 10^{-5})$$

#### Three Formal Accuracy Metrics:
1. **Raw Accuracy ($\text{Acc}_{\text{raw}}$)** (Treats ties as losses / non-wins):
   $$\text{Acc}_{\text{raw}}(h) = \frac{\sum_{t=1}^N \mathbb{I}(\text{Edge}_t(h) > 10^{-5})}{N}$$
2. **Non-Tie Accuracy ($\text{Acc}_{\text{non-tie}}$)** (Treats ties as no-trade):
   $$\text{Acc}_{\text{non-tie}}(h) = \frac{\sum_{t=1}^N \mathbb{I}(\text{Edge}_t(h) > 10^{-5})}{N - \sum_{t=1}^N \text{Tie}_t(h)}$$
3. **Latency-Adjusted Accuracy ($\text{Acc}_{\text{lat}}$)** (Execution delayed to $t+1$ due to feature computation + model inference turnaround time $\approx 1,018\text{ ms}$):
   $$R_t^{\text{lat}}(h) = \left( \frac{P_{\text{mid}, t+1+h} - P_{\text{mid}, t+1}}{P_{\text{mid}, t+1}} \right) \times 10,000$$
   $$\text{Edge}_t^{\text{lat}}(h) = \begin{cases} +R_t^{\text{lat}}(h), & \text{if } \hat{y}_t = 2 \\ -R_t^{\text{lat}}(h), & \text{if } \hat{y}_t = 0 \end{cases}$$
   $$\text{Acc}_{\text{lat}}(h) = \frac{\sum_{t=1}^{N_{\text{lat}}} \mathbb{I}(\text{Edge}_t^{\text{lat}}(h) > 10^{-5})}{N_{\text{lat}} - \text{Ties}^{\text{lat}}(h)}$$

### 3.2 Signal Clustering Algorithm & Independent Run Formulation
Signals emitted on consecutive 10-second bars share overlapping forward evaluation windows (e.g., bar 161 and bar 162 both evaluate price paths through bar 164 and 165). They violate the independent and identically distributed (i.i.d.) assumption of Bernoulli trials.

#### Clustering Algorithm:
Let $S = \{(t_i, \hat{y}_{t_i})\}_{i=1}^M$ be the sequence of directional bars ordered by time.
A run $C_k$ is formed by maximal contiguous indices where $t_{i+1} = t_i + 1$ and $\hat{y}_{t_{i+1}} = \hat{y}_{t_i}$.
Only the initial bar of each cluster $(t_{k, \text{start}}, \hat{y}_{k})$ is scored:
$$\text{RunEdge}_k(h) = \text{Edge}_{t_{k, \text{start}}}(h)$$

#### Empirical Clustering Results:
The 28 directional bars collapse into **18 independent runs**:
- 13 single-bar runs (bars 16, 39, 53, 77, 95, 99, 104, 180, 187, 191, 229, 234)
- 3 two-bar runs (bars [7, 8], [64, 65], [146, 147], [140, 141])
- 1 three-bar run (bars [59, 60, 61])
- 1 five-bar run (bars [161, 162, 163, 164, 165])

#### Run-Level Statistical Performance at Horizon H3:
- Evaluated Runs: 18 runs
- Zero-Move Ties: 4 runs ($22.2\%$)
- Active Decisive Wins / Losses: 9 Wins / 5 Losses
- **Raw Run Accuracy:** $50.0\%$ (9 / 18), Wilson 95% CI: $[29.0\%, 71.0\%]$, Binomial $p = 1.000$.
- **Non-Tie Run Accuracy:** $64.3\%$ (9 / 14), Wilson 95% CI: $[38.8\%, 83.7\%]$, Binomial $p = 0.424$.

### 3.3 Statistical Significance: Exact Binomial Test & Wilson Score Intervals
1. **Exact Two-Tailed Binomial Test ($H_0: p = 0.50$):**
   $$p\text{-value} = 2 \times \min\left( \sum_{j=0}^{k} \binom{n}{j} 0.5^n, \sum_{j=k}^{n} \binom{n}{j} 0.5^n \right)$$
2. **Wilson 95% Score Confidence Interval:**
   With $z = 1.95996$ and $\hat{p} = k/n$:
   $$\text{CI}_{\text{Wilson}} = \frac{\hat{p} + \frac{z^2}{2n} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$
3. **Required Sample Size for Validation (Statistical Power Equation):**
   To reject $H_0: p = 0.50$ at $\alpha = 0.05$ with statistical power $1 - \beta = 0.80$ ($z_{\alpha/2} = 1.96, z_{\beta} = 0.8416$):
   $$N \approx \frac{(z_{\alpha/2} + z_{\beta})^2 \cdot p(1-p)}{(p - 0.50)^2}$$
   - For a true win rate of $p = 0.55$: **$N \approx 784$ independent signal runs**.
   - For a true win rate of $p = 0.60$: **$N \approx 196$ independent signal runs**.
   - *Requirement AC4 specifies:* Validation must span **$\ge 200$ independent signal clusters** to ensure the lower bound of the Wilson 95% CI exceeds $50.0\%$.

### 3.4 Market Drift De-Biasing & Drift-Adjusted Alpha
When a model exhibits directional asymmetry (e.g., $75\%$ short signals) during a secular trend ($+25.18\text{ bps}$ over 270 bars), raw returns conflate model skill with macro market drift.
$$\text{Drift}_{\text{bar}} = \frac{P_{\text{mid}, 270} - P_{\text{mid}, 1}}{P_{\text{mid}, 1} \times 270} \times 10,000 = +0.0933\text{ bps/bar}$$
$$\text{Drift}(h) = h \times \text{Drift}_{\text{bar}}$$

Let $w_{\text{long}} = \frac{N_{\text{long}}}{N} = 0.25$ and $w_{\text{short}} = \frac{N_{\text{short}}}{N} = 0.75$.
The net macro drift headwind experienced by the strategy is:
$$\text{Headwind}(h) = (w_{\text{long}} - w_{\text{short}}) \times \text{Drift}(h) = (0.25 - 0.75) \times \text{Drift}(h) = -0.50 \times \text{Drift}(h)$$
The drift-adjusted alpha edge is:
$$\text{Edge}_{\text{drift-adj}}(h) = \text{Edge}_{\text{raw}}(h) - \text{Headwind}(h) = \text{Edge}_{\text{raw}}(h) + 0.50 \times \text{Drift}(h)$$

- **At H1 (10s):** Raw $+0.245\text{ bps}$ | Drift Headwind $-0.047\text{ bps}$ | **Drift-Adjusted: $+0.292\text{ bps}$**
- **At H2 (20s):** Raw $+0.236\text{ bps}$ | Drift Headwind $-0.093\text{ bps}$ | **Drift-Adjusted: $+0.329\text{ bps}$**
- **At H3 (30s):** Raw $+0.227\text{ bps}$ | Drift Headwind $-0.140\text{ bps}$ | **Drift-Adjusted: $+0.367\text{ bps}$ (Peak Alpha)**
- **At H6 (60s):** Raw $-0.125\text{ bps}$ | Drift Headwind $-0.280\text{ bps}$ | **Drift-Adjusted: $+0.155\text{ bps}$**
- **At H12 (120s):** Raw $-0.562\text{ bps}$ | Drift Headwind $-0.560\text{ bps}$ | **Drift-Adjusted: $-0.002\text{ bps}$**
- **At H18 (180s):** Raw $+0.034\text{ bps}$ | Drift Headwind $-0.839\text{ bps}$ | **Drift-Adjusted: $+0.873\text{ bps}$**

### 3.5 Brier Score & Expected Calibration Error (ECE)
1. **Multi-Class Brier Score:**
   $$\text{Brier} = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^3 (\hat{p}_{i,k} - y_{i,k})^2$$
   where $y_{i,k} \in \{0, 1\}$ is the one-hot target defined by a return deadband $\theta = 1.5\text{ bps}$:
   $$y_i = \begin{cases} [1, 0, 0] \text{ (DOWN)}, & \text{if } R_i(H3) < -1.5\text{ bps} \\ [0, 1, 0] \text{ (FLAT)}, & \text{if } |R_i(H3)| \le 1.5\text{ bps} \\ [0, 0, 1] \text{ (UP)}, & \text{if } R_i(H3) > +1.5\text{ bps} \end{cases}$$
   - **Climatological Prior Baseline:** A naive model assigning fixed prior class frequencies $[P_{\text{down}}=0.078, P_{\text{flat}}=0.896, P_{\text{up}}=0.026]$ achieves:
     $$\text{Brier}_{\text{clim}} = \mathbf{0.2786}$$
   - **Uniform Baseline:** $[1/3, 1/3, 1/3] \implies \text{Brier} = \mathbf{0.6667}$.
   - **Legacy Ensemble Score:** $\text{Brier}_{\text{model}} = \mathbf{0.4165}$ (severely uncalibrated, worse than climatology).
   - **Acceptance Criterion AC2:** $\text{Brier}_{\text{new}} < \mathbf{0.2786}$.

2. **Expected Calibration Error (ECE):**
   Group predicted maximum confidence $\hat{c}_i = \max_k \hat{p}_{i,k}$ into $M=5$ bins $B_m$:
   $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$$
   - **Legacy ECE Measured:** **$20.74\%$**.
   - Pathological Failure: In the $[0.70, 0.80)$ bin ($N=5$), mean predicted confidence was $74.36\%$, but empirical accuracy was $0.0\%$ (gap $= -74.36\%$).
   - **Acceptance Criterion AC2:** $\text{ECE} \le \mathbf{8.0\%}$.

### 3.6 Transaction Fee Cost Structure (Binance USD-M VIP0)
- **Maker Fee:** $0.020\%$ per leg ($2.0\text{ bps}$).
  $$\text{Fee}_{\text{RT, Maker}} = 2.0\text{ bps (entry)} + 2.0\text{ bps (exit)} = \mathbf{4.0\text{ bps}}$$
- **Taker Fee:** $0.050\%$ per leg ($5.0\text{ bps}$).
  $$\text{Fee}_{\text{RT, Taker}} = 5.0\text{ bps (entry)} + 5.0\text{ bps (exit)} = \mathbf{10.0\text{ bps}}$$
- **Maker Advantage:** Earning maker execution saves $6.0\text{ bps}$ round-trip ($60\%$ fee reduction).

### 3.7 Volatility Hurdles & Move-Magnitude Thresholds
- **Realized Volatility Gate:** Reject all trade authorizations if 15-minute rolling realized volatility is low:
  $$\sigma_{15\text{m}} < 15.0\text{ bps} \quad \text{or} \quad \text{Mean}|R_{10\text{s}}| < 1.0\text{ bps}$$
- **Expected Move Magnitude Hurdle:** A trade is authorized only if the model's predicted move magnitude $|\Delta \hat{p}|$ satisfies:
  $$|\Delta \hat{p}| \ge \text{Fee}_{\text{RT, Maker}} + \text{Buffer} = 4.0\text{ bps} + 2.0\text{ bps} = \mathbf{6.0\text{ bps}}$$
- **Dynamic ATR Barriers:**
  $$\text{TP} = k_{\text{tp}} \times \text{ATR}_{10\text{s}}, \quad \text{SL} = k_{\text{sl}} \times \text{ATR}_{10\text{s}}$$
  Replacing static $+16.0 / -12.0\text{ bps}$ with volatility-scaled values (typically $2.0 - 4.5\text{ bps}$ in quiet markets).

---

## 4. Features Discovered & Probed

### 4.1 Features Discovered Table

| # | Category | Feature | Description | Inputs | Outputs | Error Behavior | Discovered Via |
|---|----------|---------|-------------|--------|---------|----------------|----------------|
| 1 | Microstructure | `bbo_mid` | Midpoint between best bid and best ask on Binance USD-M futures | `best_bid`, `best_ask` | Float (BBO mid-price) | Fallback to `close_price` if bid/ask missing | `scratch/live_45m_eval_ledger.jsonl`, `rigorous_evaluation_audit.py` |
| 2 | Telemetry / Eval | `forward_returns_bps` | Forward mid-price return in bps at horizons H1, H2, H3, H6, H12, H18 | `bbo_mid[t]`, `bbo_mid[t+h]` | Dict `{str(h): float}` | Omitted for last $h$ bars of session | `scratch/live_45m_eval_ledger.jsonl`, `live_45m_comprehensive_eval.py` |
| 3 | Telemetry / Eval | `forward_returns_latency_bps` | Latency-adjusted forward return entering at $t+1$ and exiting at $t+1+h$ | `bbo_mid[t+1]`, `bbo_mid[t+1+h]` | Dict `{str(h): float}` | Omitted for last $h+1$ bars | `scratch/live_45m_eval_ledger.jsonl`, `rigorous_evaluation_audit.py` |
| 4 | Microstructure | `order_flow_imbalance_near` | Normalized L1 order flow imbalance pressure | Best bid/ask size changes | Float in $[-1.0, 1.0]$ | Clipped to $[-1.0, 1.0]$ | `checkpoint_loaders.py`, `feature_contract.py` |
| 5 | Microstructure | `taker_buy_sell_imbalance` | Volume-weighted aggressive taker buy vs sell imbalance | Taker trade volumes | Float in $[-1.0, 1.0]$ | Clipped to $[-1.0, 1.0]$ | `checkpoint_loaders.py`, `feature_contract.py` |
| 6 | Flow / Leakage | `_live_ret_pct` | Intra-bar price change injected by live WebSocket ingestor | Real-time WS ticker | Float (%) | Leaks intra-bar price return after close boundary | `checkpoint_loaders.py:115` |
| 7 | Meta-Labeling | `act_probability` (legacy) | Meta-labeler probability that primary bet will be decisive | `p_sel`, `p_flat`, `conf` | Float in $[0.30, 0.96]$ | Clamped to $0.42$ on flat; bimodal jump across $0.52-0.85$ | `live_loop_v2.py:189-196` |
| 8 | Execution / Sizing | `bet_size` (legacy) | Position sizing scalar multiplier | `act_probability`, `confidence` | Float in $[0.0, 1.0]$ | Zeroed out if `act_probability < 0.70` | `live_loop_v2.py:198` |
| 9 | Statistical Eval | `wilson_ci` | Wilson score 95% confidence interval for binomial proportion | Wins $k$, trials $n$ | Tuple `(low, high, p_val)` | Returns `(0, 0, 0)` if $n=0$ | `rigorous_evaluation_audit.py:32` |
| 10 | Statistical Eval | `cluster_runs` | Grouping consecutive identical directional signals into independent trials | Ordered signal sequence | List of bar index clusters | Empty list if no directional signals | `rigorous_evaluation_audit.py:141` |
| 11 | Statistical Eval | `drift_adjusted_alpha` | Net return edge controlling for secular trend headwind | Raw edge, market drift, short/long ratio | Float (bps) | None | `rigorous_evaluation_audit.py:180` |
| 12 | Calibration Eval | `expected_calibration_error` | Binned absolute difference between mean confidence and accuracy | Model probabilities, true labels | Float (0.0 to 1.0) | Empty bins skipped | `rigorous_evaluation_audit.py:273` |
| 13 | Cross-Asset Alpha | Lead-lag cross-asset (R1 target) | Lead-lag dynamics between ETHUSDT/SOLUSDT and BTCUSDT | Multi-asset order book flow | Continuous alpha score | TBD in M1 | `ORIGINAL_REQUEST.md:31` |
| 14 | Funding / Basis | Spot-perpetual basis divergence (R1 target)| Premium/discount between spot BTC and USD-M perpetual | Spot BBO, Perp BBO | Continuous bps spread | TBD in M1 | `ORIGINAL_REQUEST.md:32` |
| 15 | Microstructure | VPIN / Liquidation burst (R1 target) | Volume-synchronized probability of toxicity & liquidation flow | Trade flow, liquidation stream | Probability score | TBD in M1 | `ORIGINAL_REQUEST.md:33` |
| 16 | Execution | BBO Queue-Aware Maker Node (R3 target) | Passive limit orders posted at BBO using `GTX` with queue tracking | Order book L1/L2 depth | Order events, cancel/replaces | Adverse selection abort | `ORIGINAL_REQUEST.md:45-46` |

---

## 5. Edge Cases Discovered

| # | Feature | Input | Observed Behavior |
|---|---------|-------|-------------------|
| 1 | Directional Return ($H1$) | Zero price move ($\Delta P = 0.00$) | Exactly **$42.86\%$ (12/28)** of 10s bars have zero move. Scoring as loss causes 46.4% accuracy; scoring as no-trade causes 81.2% accuracy. |
| 2 | Directional Return ($H3$) | Zero price move ($\Delta P = 0.00$) | Exactly **$21.43\%$ (6/28)** of 30s bars have zero move. Scoring as loss causes 46.4% accuracy; scoring as no-trade causes 59.1% accuracy. |
| 3 | Latency Offset ($t+1018\text{ ms}$) | Entering at bar $i+1$ instead of bar $i$ | Turnaround latency flips positive gross raw edge ($+0.227\text{ bps}$) to negative ($-0.176\text{ bps}$) at $H3$, and drops $H1$ non-tie accuracy from $81.2\%$ to $44.4\%$. |
| 4 | Meta-Labeler Activation | `p_sel < 0.62` or `p_flat > 0.28` | Output clamped to $[0.30, 0.52]$ (empirically $0.380 - 0.473$). |
| 5 | Meta-Labeler Activation | `p_sel >= 0.62` and `p_flat <= 0.28` | Formula creates a mathematical discontinuity jumping immediately to $[0.850, 0.942]$. Zero samples fall between $0.52$ and $0.85$. |
| 6 | Meta-Labeler Activation | `primary_direction == 1` (FLAT) | Exactly $242$ bars ($89.6\%$) receive a constant hardcoded value of $0.420$. |
| 7 | High-Conviction Filter | Thresholds $\tau = 0.50, 0.60, 0.70$ | All three thresholds produce the exact same 5 bars because of the empty dead zone between $0.52$ and $0.85$. |
| 8 | Fixed Triple Barrier (180s) | Static $+16.0\text{ bps}$ TP barrier | 100% of trades failed to reach the barrier (mean move was only $0.74\text{ bps}$), expiring at max hold (180s) with 100% time-stop loss. |
| 9 | Model Inter-Correlation | Raw predictions of `tft`, `patchtst`, `nhits`, `ssm` | Pairwise correlations are $0.987$ to $0.997$ due to shared reliance on `_live_micro_signal` and identical underlying features. |
| 10 | Sentiment Branch Feature Extraction | `_live_ret_pct` present in `recent_df` | `_live_micro_signal` consumes intra-bar price movement after the bar close boundary, introducing subtle lookahead bias. |

---

## 6. Complete Requirements & Acceptance Criteria Inventory

### R1. Orthogonal Multi-Architecture Ensemble & Feature De-Duplication
- **De-correlate Member Predictions:**
  - *Current State:* NHITS, PatchTST, TFT, and SSM correlate at $r = 0.987 - 0.997$.
  - *Requirement:* Replace redundant models with diverse signal generators.
  - *Threshold (AC1):* Pairwise Pearson and Spearman correlation of raw directional predictions between any two members $\le 0.70$ across the validation set.
  - *Signal Families Required (AC1):* At least 3 distinct feature families:
    1. Lead-lag cross-asset dynamics (e.g. ETHUSDT / SOLUSDT order book flow predicting BTCUSDT).
    2. Spot-perpetual basis divergence and funding rate pressure.
    3. Order book microstructure: L1 queue imbalance, microprice drift, trade flow imbalance (VPIN / toxicity), and large liquidation bursts.
- **Strict Time-Series Integrity & Anti-Leakage:**
  - *Threshold (AC3):* 100% of feature inputs at bar $t$ must strictly use information timestamped $\le T_{\text{close}} - 1,000\text{ ms}$.
  - *Sentiment Branch Fix (AC3):* Zero intra-bar or future price return leakage in `_live_micro_signal` / `sentiment-branch`.
- **Dimensionality & Regularization:**
  - Prune the 174-feature set down to high signal-to-noise orthogonal features using mutual information or recursive feature elimination with purged walk-forward cross-validation.

### R2. Continuous Calibrated Meta-Labeler & Volatility Hurdle Engine
- **Continuous Probabilistic Meta-Labeling:**
  - *Threshold (AC2):* Produce unconstrained, smoothly calibrated confidence probabilities $\hat{p} \in [0, 1]$ across the full continuous range. Completely remove artificial piecewise clamps (e.g. constant $0.42$ and $[0.52, 0.85]$ dead zones).
- **Probability Calibration:**
  - *Threshold (AC2):* Expected Calibration Error (ECE) $\le 8.0\%$ on out-of-fold validation data.
  - *Threshold (AC2):* Brier score strictly lower than the climatological prior baseline ($< 0.2786$).
- **Cost & Magnitude Hurdle:**
  - Trade authorization condition: $|\Delta \hat{p}| \ge \text{Fee}_{\text{round-trip}} + \text{Buffer} \ge 4.0\text{ bps} + 2.0\text{ bps} = 6.0\text{ bps}$.
- **Realized Volatility Gate:**
  - Halt directional authorizations when 10-second mean absolute move $< 1.0\text{ bps}$ or 15-minute realized volatility $< 15.0\text{ bps}$.

### R3. BBO Queue-Aware Maker Execution Node & Barrier Synchronization
- **Horizon & Barrier Synchronization:**
  - *Threshold (AC4):* Align position holding horizon strictly with the empirical alpha decay curve: holding horizon $= 3\text{ bars} / 30\text{ seconds}$ (trade duration $\le 30\text{ seconds}$).
  - Replace static $+16/-12\text{ bps}$ barriers with dynamic volatility-scaled barriers ($k \times \text{ATR}_{10\text{s}}$).
- **Pure Maker-Post Execution:**
  - Formulate execution as limit order placement at BBO using `GTX` (Post-Only) order types to capture Binance USD-M VIP0 maker fee ($2.0\text{ bps}$ per leg, $4.0\text{ bps}$ round-trip vs $10.0\text{ bps}$ taker).
- **Queue Priority & Adverse Selection Modeling:**
  - Track queue position and cancel/reprice immediately if queue or order book imbalance deteriorates.

### R4. Institutional Purged Walk-Forward Evaluation & Live Telemetry Harness
- **Purged Walk-Forward Cross-Validation:**
  - Rolling out-of-sample windows with strict purging (dropping overlapping test return windows) and embargo ($\ge 3$ bars).
- **Clustered Run & Confidence Metrics (AC4):**
  - Spanning $\ge 200$ independent signal clusters across multiple volatility regimes.
  - Positive net PnL after Binance USD-M VIP0 maker fees ($4.0\text{ bps}$ round-trip).
  - Wilson 95% confidence interval lower bound for directional accuracy $> 50.0\%$.
  - Average trade duration $\le 30\text{ seconds}$.
- **Continuous Telemetry Ledger:**
  - Stream all live signals, timestamps, BBO bid/ask/mid, fill events, and forward returns ($H1, H2, H3, H6, H12, H18$) to persistent disk ledgers.
- **Reproducible Test Suite & Inference Latency (AC5):**
  - Automated test suite validating data integrity, feature alignment, and execution logic.
  - End-to-end model inference latency $\le 250\text{ ms}$.
  - Unified evaluation script reproducing all metrics directly from the telemetry ledger.

---

## 7. Verification Methodology & Harness Commands

The four authoritative verification gates prescribed by the project specification:
```powershell
# 1. Run Unit & Data Leakage Tests (Verifies AC3: zero leakage, t <= T_close - 1000ms)
pytest tests/test_leakage_and_features.py -v

# 2. Run Ensemble Orthogonality & Calibration Evaluation (Verifies AC1 & AC2: r <= 0.70, ECE <= 8.0%, Brier < 0.2786)
python scripts/evaluate_ensemble_and_meta.py --data-dir "D:/backup/Binance data"

# 3. Run Purged Walk-Forward Simulation (Net of 4 bps Maker Fees) (Verifies AC4: >= 200 runs, net PnL > 0, Wilson low > 50%, hold <= 30s)
python scripts/purged_walk_forward_backtest.py --fees-bps 4.0 --embargo-bars 3

# 4. Verify Telemetry & Ledger Rigor (Verifies AC5: reproduces all statistical metrics from raw ledger)
python "feature engineering scripts/scratch/rigorous_evaluation_audit.py"
```

---

## 8. Summary of Engineering Failure Modes & Remediation Roadmap

1. **Failure Mode 1: Intra-Bar Lookahead Leakage**
   - *Root Cause:* `_live_micro_signal()` in `checkpoint_loaders.py` injected unclosed bar returns (`_live_ret_pct`).
   - *Remediation:* Enforce strict cutoff $t \le T_{\text{close}} - 1,000\text{ ms}$; purge intra-bar columns.
2. **Failure Mode 2: Extreme Model Co-linearity**
   - *Root Cause:* Transformer models (NHITS, PatchTST, TFT, SSM) were trained on identical features and shared the same `_live_micro_signal` wrapper, correlating at $0.987 - 0.997$.
   - *Remediation:* Prune collinear models; replace with genuinely distinct feature families (Cross-Asset lead-lag, Spot-Perp basis, L1 queue imbalance/VPIN).
3. **Failure Mode 3: Meta-Labeler Bimodal Collapse**
   - *Root Cause:* Hardcoded piecewise branching in `live_loop_v2.py` clamped flat bars to $0.42$ and split directional bars into $[0.38, 0.47]$ vs $[0.85, 0.94]$.
   - *Remediation:* Train continuous secondary classifier with Platt scaling / isotonic regression on OOF data.
4. **Failure Mode 4: Horizon & Barrier Mismatch**
   - *Root Cause:* Offline labels used 360 bars (1 hr); previous execution used 18 bars (180s) with unreachable $+16/-12\text{ bps}$ static barriers; while empirical alpha peaks at 3 bars (30s) with mean move $\approx 0.74\text{ bps}$.
   - *Remediation:* Synchronize execution holding time to 3 bars (30s); replace static barriers with dynamic $k \times \text{ATR}_{10\text{s}}$ barriers; require $|\Delta \hat{p}| \ge 6.0\text{ bps}$.
5. **Failure Mode 5: Execution Fee Annihilation**
   - *Root Cause:* Taking liquidity at 10 bps RT completely wiped out small alpha moves.
   - *Remediation:* Pure maker `GTX` BBO queue-aware execution node capturing VIP0 4.0 bps RT rate.
