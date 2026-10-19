# Handoff Report — Spec & Ledger Mining Survey

**Author:** Spec & Ledger Miner (`teamwork_preview_spec_miner`)  
**Working Directory:** `D:\backup\Binance data\.agents\teamwork\spec_miner_survey`  
**Recipient:** Project Orchestrator (`4a8e5400-290d-44ec-9df4-e3142d1475c4`)  
**Date:** 2026-10-18T01:27:00Z  
**Type:** Hard Handoff (Task Complete)

---

## 1. Observation

Direct observations from the authoritative sources and raw datasets in `D:\backup\Binance data`:

1. **Empirical Ledger Data (`scratch/live_45m_eval_ledger.jsonl`)**:
   - 270 ten-second bars spanning 48.01 minutes (2026-10-17 17:54:40 to 18:42:41 UTC).
   - Market drift: BTC BBO mid moved from $\$81,329.05$ to $\$81,533.85$ ($+25.18\text{ bps}$, $+0.0933\text{ bps/bar}$). 66.3% of 180s rolling windows ended positive.
   - Class distribution: 242 FLAT bars ($89.63\%$), 28 directional bars ($10.37\%$: 21 SHORT, 7 LONG).
   - Zero-move bars: Exactly 12 out of 28 directional bars ($42.86\%$) at H1 (10s) and 6 out of 28 ($21.43\%$) at H3 (30s) had zero price movement ($0.00\text{ bps}$).

2. **Empirical Directional Accuracy & P-Values (`result.md:57-65`, `rigorous_evaluation_audit.py:90-138`)**:
   - At H3 (30s / 3 bars, pre-registered primary horizon):
     - Raw Hit Rate: $46.43\%$ (13/28), Wilson 95% CI: $[29.5\%, 64.2\%]$, Binomial $p = 0.851$.
     - Non-Tie Hit Rate: $59.09\%$ (13/22), Wilson 95% CI: $[38.7\%, 76.7\%]$, Binomial $p = 0.523$.
     - Latency-Adjusted Hit Rate ($t+1018\text{ ms}$): $52.94\%$ (9/17), Wilson 95% CI: $[31.0\%, 73.8\%]$, Binomial $p = 1.000$.
     - Gross edge at H3: $+0.227\text{ bps}$ raw, $-0.176\text{ bps}$ latency-adjusted.

3. **Clustered Runs (`result.md:75-105`, `rigorous_evaluation_audit.py:140-178`)**:
   - 28 directional bars collapse into 18 independent contiguous-direction runs.
   - Run-level raw accuracy at H3: $50.0\%$ (9/18), Wilson 95% CI: $[29.0\%, 71.0\%]$, $p = 1.000$.
   - Run-level non-tie accuracy at H3: $64.3\%$ (9/14), Wilson 95% CI: $[38.8\%, 83.7\%]$, $p = 0.424$.

4. **Baselines & Calibration (`result.md:144-185`, `rigorous_evaluation_audit.py:192-300`)**:
   - Always-LONG baseline edge at H3: $+0.614\text{ bps}$ (outperformed ensemble $+0.227\text{ bps}$).
   - Climatological Brier score: $0.2786$ vs Ensemble Brier score: $0.4165$ (Uniform: $0.6667$).
   - Expected Calibration Error (ECE): $20.74\%$. Pathological failure in $[0.70, 0.80)$ bin ($N=5$) where empirical accuracy was $0.0\%$ (mean confidence $74.36\%$, gap $-74.36\%$).

5. **Meta-Labeler Distribution Collapse (`live_loop_v2.py:189-201`, `result.md:190-204`)**:
   - Lines 189-196 implement a piecewise clamp: `p_sel < 0.62 or p_flat > 0.28` forces `act_probability` into $[0.30, 0.52]$; else jumps to $[0.85, 0.94]$.
   - All 242 flat bars were assigned constant $0.420$. Zero bars fall in $(0.52, 0.85)$.
   - The 5 authorized bars ($\tau \ge 0.70$) all lost net under taker (-10 bps) and maker (-4 bps).

6. **Model Collinearity (`result.md:225-244`, `rigorous_evaluation_audit.py:334-341`)**:
   - Pairwise correlations: `tft`, `patchtst`, `nhits`, and `ssm-challenger` correlate at $0.987 - 0.997$.
   - Source: `checkpoint_loaders.py:194, 276, 359, 488, 593` blends identical `_live_micro_signal(recent_df)` (30%-40% weight) across all members.
   - `checkpoint_loaders.py:115-119` incorporates unclosed bar returns `_live_ret_pct`, creating intra-bar forward leakage.

7. **Horizon Mismatch (`config.py:140`, `live_45m_comprehensive_eval.py:42-47`)**:
   - `TRIPLE_BARRIER_MAX_HOLD_BARS` in config: 360 bars (1 hr).
   - Execution in `live_45m_comprehensive_eval.py`: 18 bars (180s), TP $+16\text{ bps}$, SL $-12\text{ bps}$.
   - Empirical alpha peaks at 20-30s (3 bars) with mean move $\approx 0.74\text{ bps}$. 100% of trades expired via time-stop.

---

## 2. Logic Chain

1. **From Observation 1 & 2 to Hit Rate Collapse:**
   Scoring against BBO mid-price eliminates bid-ask bounce. Zero-move microstructure ties constitute $42.9\%$ of 10s bars and $21.4\%$ of 30s bars. At the pre-registered 30-second primary horizon ($H3$), raw accuracy is $46.4\%$ ($p = 0.851$) and non-tie accuracy is $59.1\%$ ($p = 0.523$). Neither is statistically different from random noise.
2. **From Observation 2 to Latency Sensitivity:**
   Turnaround latency ($t_{\text{latency}} \approx 1,018\text{ ms}$) shifts execution to bar $i+1$, immediately dropping $H1$ non-tie hit rate from $81.2\%$ to $44.4\%$ and inverting $H3$ return from $+0.227\text{ bps}$ to $-0.176\text{ bps}$.
3. **From Observation 3 to Sample Size Deficit:**
   Consecutive same-direction signals share forward paths and must be clustered into runs (18 runs). At the run level, raw hit rate is $50.0\%$ ($p = 1.000$). Power analysis proves that validating a $55\%$ win rate requires $N \approx 784$ independent runs; validating a $60\%$ win rate requires $N \approx 196$. 18 runs is underpowered by an order of magnitude.
4. **From Observation 4 to Calibration Failure:**
   The ensemble's Brier score of $0.4165$ is severely worse than the naive climatological prior ($0.2786$) because it overconfidently assigns probability mass to directional tails that fail to materialize in chop regimes ($89.6\%$ flat). ECE is $20.74\%$ with $0\%$ accuracy in the highest confidence bucket.
5. **From Observation 5 to Meta-Labeler Distortion:**
   The piecewise logic in `live_loop_v2.py` clamped flat bars to $0.420$ and created a dead zone between $0.52$ and $0.85$, turning $\tau = 0.50, 0.60, 0.70$ into identical filters selecting only 5 bars.
6. **From Observation 6 to Redundancy & Leakage:**
   All 4 transformer/SSM models correlate at $> 0.98$ because they share identical training features and a common `_live_micro_signal` wrapper that samples unclosed intra-bar returns (`_live_ret_pct`).
7. **From Observation 7 to Execution Barrier Failure:**
   Setting a $+16\text{ bps}$ take-profit barrier when empirical 30s alpha is $\approx 0.74\text{ bps}$ guarantees $100\%$ time-stop exits. Holding for 180s into macro drift caused 100% gross and net losses.

---

## 3. Caveats

1. The live telemetry dataset comprises 270 bars (48 minutes) containing 18 independent directional signal runs. While sufficient to diagnose structural pipeline flaws, statistical power to prove a true positive edge requires $\ge 200$ independent clusters under purged walk-forward cross-validation (as mandated by AC4).
2. The Binance USD-M VIP0 maker fee of $2.0\text{ bps}$ per leg ($4.0\text{ bps}$ round-trip) assumes passive execution via `GTX` limit orders at the BBO; it does not model queue adverse selection or replenishment queue slippage, which must be addressed in Milestone 3.

---

## 4. Conclusion

The specification survey is complete. The exact statistical formulas, empirical numbers, requirement thresholds, and verification commands have been extracted, formalized, and written to:
`D:\backup\Binance data\.agents\teamwork\spec_miner_survey\survey_spec_report.md`

### Key Numerical Thresholds Established for Downstream Workers:
- **Pairwise Ensemble Correlation (AC1):** $\le 0.70$ across all member pairs.
- **Continuous Meta-Labeler Calibration (AC2):** Continuous $\hat{p} \in [0, 1]$, $\text{ECE} \le 8.0\%$, $\text{Brier} < 0.2786$.
- **Anti-Leakage Cutoff (AC3):** $100\%$ features timestamped $\le T_{\text{close}} - 1,000\text{ ms}$; zero intra-bar returns.
- **Purged Walk-Forward Performance (AC4):** $\ge 200$ independent signal clusters; positive net PnL net of $4.0\text{ bps}$ maker fees; Wilson 95% CI lower bound $> 50.0\%$; holding horizon $\le 30\text{ seconds}$.
- **Hurdles & Gates:** Predicted move $|\Delta \hat{p}| \ge 6.0\text{ bps}$; chop gate halts if $\text{Mean}|R_{10\text{s}}| < 1.0\text{ bps}$ or $\sigma_{15\text{m}} < 15.0\text{ bps}$.
- **Inference Latency (AC5):** Total turnaround $\le 250\text{ ms}$.

---

## 5. Verification Method

To independently verify the empirical numbers and statistical formulations documented in this report:

1. **Inspect Survey Report:**
   ```powershell
   Get-Content "D:\backup\Binance data\.agents\teamwork\spec_miner_survey\survey_spec_report.md"
   ```
2. **Inspect Quantitative Audit Ledger & Output:**
   - Raw Ledger: `D:\backup\Binance data\scratch\live_45m_eval_ledger.jsonl`
   - Official Audit Report: `D:\backup\Binance data\result.md`
   - Audit Implementation: `D:\backup\Binance data\feature engineering scripts\scratch\rigorous_evaluation_audit.py`
3. **Invalidation Conditions:**
   - Any discrepancy between reported CI/p-values and `result.md` (e.g., $H3$ raw hit rate $\ne 46.43\%$, non-tie $\ne 59.09\%$, p-value $\ne 0.523$, run count $\ne 18$).
   - Any claim of member decorrelation prior to eliminating `_live_micro_signal`.
