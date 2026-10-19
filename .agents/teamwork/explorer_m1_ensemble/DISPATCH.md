## 2026-10-18T01:31:36Z
You are the Ensemble De-Correlation Explorer for Milestone 1.

Your Identity & Working Directory:
- Working Directory: D:\backup\Binance data\.agents\teamwork\explorer_m1_ensemble
- Parent Conversation ID: 4a8e5400-290d-44ec-9df4-e3142d1475c4

MANDATORY FIRST STEP:
Read both:
1. D:\backup\Binance data\.agents\teamwork\ORIGINAL_REQUEST.md
2. D:\backup\Binance data\PROJECT.md

Your Mission:
Formulate the exact refactoring strategy to break the collinearity among ensemble member predictors (currently r > 0.98) and achieve pairwise r <= 0.70 across the validation set (AC1):
1. Deeply inspect `feature engineering scripts/inference/checkpoint_loaders.py`, `fusion/oof/`, and `streaming/live_loop_v2.py`.
2. Analyze why offline models had low correlation (r = 0.05 - 0.52) while live predictions collapsed to r > 0.98 (shared `_live_micro_signal` blending, identical wrapper logic, uniform sigmoid mapping).
3. Design 3 diverse, specialized predictor modules:
   - Model A (Microstructure Specialist): Focuses on L1 queue imbalance, OFI, VPIN, microprice.
   - Model B (Cross-Asset Specialist): Focuses on ETH lead-lag flow and cross-asset relative momentum.
   - Model C (Basis & Term Structure Specialist): Focuses on Spot-Perp basis divergence, funding rates, and multi-horizon trend.
4. Specify clean inference contracts, distinct feature routing, and the removal of artificial shared wrappers.
5. Detail how pairwise Pearson and Spearman correlation will be evaluated to strictly satisfy AC1 (r <= 0.70).
6. Write your strategy report to `D:\backup\Binance data\.agents\teamwork\explorer_m1_ensemble\m1_ensemble_strategy.md` and deliver `handoff.md`.
7. Send a completion message back to parent (4a8e5400-290d-44ec-9df4-e3142d1475c4).

Constraints: Read-only exploration. Do NOT modify source files directly. Provide complete architecture for the Worker.
