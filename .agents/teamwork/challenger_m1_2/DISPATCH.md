## 2026-10-18T01:55:38Z
You are Challenger 2 (De-Correlation Stress Verifier) for Milestone 1.

Your Identity & Working Directory:
- Working Directory: D:\backup\Binance data\.agents\teamwork\challenger_m1_2
- Parent Conversation ID: 4a8e5400-290d-44ec-9df4-e3142d1475c4

MANDATORY FIRST STEPS:
1. Read D:\backup\Binance data\.agents\teamwork\ORIGINAL_REQUEST.md
2. Read D:\backup\Binance data\PROJECT.md
3. Read the worker handoff: D:\backup\Binance data\.agents\teamwork\worker_m1\handoff.md

Your Mission:
Empirically and adversarially stress-test the ensemble de-correlation claims (AC1):
1. Execute `scripts/evaluate_ensemble_and_meta.py --data-dir "D:/backup/Binance data"` across different slices or subsets of data (e.g. high volatility vs low volatility, trends vs chop).
2. Stress test the 3 specialized member predictors (`MicrostructureSpecialist`, `CrossAssetSpecialist`, `BasisTermStructureSpecialist`) to confirm that pairwise Pearson and Spearman correlation remains strictly <= 0.70 under adverse perturbations.
3. Verify that the 3 members do not share hidden state, identical weights, or heuristic wrappers.
4. Issue an empirical correctness confirmation: CONFIRMED or DISPROVEN.
5. Write your complete verification report to `D:\backup\Binance data\.agents\teamwork\challenger_m1_2\handoff.md`.
6. Send a completion message back to parent (4a8e5400-290d-44ec-9df4-e3142d1475c4) with your findings.
