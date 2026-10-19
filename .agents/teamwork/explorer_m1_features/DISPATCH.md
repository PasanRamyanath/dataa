## 2026-10-18T01:31:35Z
You are the Orthogonal Features Explorer for Milestone 1.

Your Identity & Working Directory:
- Working Directory: D:\backup\Binance data\.agents\teamwork\explorer_m1_features
- Parent Conversation ID: 4a8e5400-290d-44ec-9df4-e3142d1475c4

MANDATORY FIRST STEP:
Read both:
1. D:\backup\Binance data\.agents\teamwork\ORIGINAL_REQUEST.md
2. D:\backup\Binance data\PROJECT.md

Your Mission:
Formulate the exact mathematical and algorithmic design for the 3 required orthogonal feature families and the feature pruning pipeline in `feature engineering scripts/features/`:
1. Design Family 1: Cross-Asset Lead-Lag Dynamics
   - ETHUSDT vs BTCUSDT lead-lag returns, order flow imbalance, and cross-asset momentum over 10s, 30s, and 60s windows.
2. Design Family 2: Spot-Perpetual Basis Divergence & Funding Pressure
   - Basis spread = (Perp_mid - Spot_mid) / Spot_mid (bps), rolling basis z-score, basis acceleration, and funding rate pressure.
3. Design Family 3: Order Book Microstructure & Flow Toxicity
   - Level 1 queue imbalance, microprice drift (Stoikov microprice), Trade Flow Imbalance / VPIN (Volume-Synchronized Probability of Toxicity), and large liquidation bursts.
4. Design the Feature Pruning Pipeline:
   - Method for reducing the current 174-feature set down to high-SNR, low-correlation orthogonal features using mutual information or recursive feature elimination with purged walk-forward CV.
5. Provide precise mathematical formulas, function signatures, input/output contracts, and fallback behaviors.
6. Write your strategy report to `D:\backup\Binance data\.agents\teamwork\explorer_m1_features\m1_features_strategy.md` and deliver `handoff.md`.
7. Send a completion message back to parent (4a8e5400-290d-44ec-9df4-e3142d1475c4).

Constraints: Read-only exploration. Do NOT modify source files directly. Provide complete formulas and implementation blueprint for the Worker.
