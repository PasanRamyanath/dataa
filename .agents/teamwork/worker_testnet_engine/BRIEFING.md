# BRIEFING — 2026-10-18T02:38:00Z

## Mission
Implement the autonomous Binance USD-M testnet execution engine (`feature engineering scripts/inference/binance_testnet_engine.py`) integrating BBOExecutionNode (GTX Post-Only, zero taker fallback, FIFO queue tracking, adverse selection cancellation, dynamic ATR barriers, 30s horizon, telemetry logging) and enhance `feature engineering scripts/scratch/rigorous_evaluation_audit.py` to support `scratch/live_streaming_eval_ledger.jsonl`.

## 🔒 My Identity
- Archetype: teamwork_preview_worker
- Roles: implementer, qa, specialist
- Working directory: D:\backup\Binance data\.agents\teamwork\worker_testnet_engine
- Original parent: 6f56ce3c-0c13-49cc-a255-279a76a73637
- Milestone: M3 (Autonomous BBO Maker Execution Node on Binance USD-M Testnet) & M4/Audit Integration

## 🔒 Key Constraints
- Exclusively modify:
  - `feature engineering scripts/inference/binance_testnet_engine.py`
  - `feature engineering scripts/scratch/rigorous_evaluation_audit.py`
- DO NOT modify files owned by other agents (`binance_ws.py`, `maker_node.py`, `live_loop_v2.py`, `tests/`, etc.).
- DO NOT cheat, fake, hardcode results or create facades.
- Pure Post-Only (GTX) limit orders with ZERO taker fallback (on -5022, cancel/abort or reprice passively, never take).
- FIFO queue tracking ($Q_{\text{ahead}}$) and adverse selection cancellation.
- Dynamic ATR barriers ($1.5 \times \text{ATR}_{10s}$ TP, $1.2 \times \text{ATR}_{10s}$ SL) and strict 30s horizon auto-flattening.
- Telemetry logging to `scratch/live_streaming_eval_ledger.jsonl`.
- Verify using:
  - `pytest tests/test_leakage_and_features.py -v`
  - `python "feature engineering scripts/scratch/rigorous_evaluation_audit.py"`
  - `python scripts/purged_walk_forward_backtest.py --fees-bps 4.0 --embargo-bars 3`
- Write handoff report to `D:\backup\Binance data\.agents\teamwork\worker_testnet_engine\handoff.md`.

## Current Parent
- Conversation ID: 6f56ce3c-0c13-49cc-a255-279a76a73637
- Updated: 2026-10-18T02:38:00Z

## Task Summary
- **What to build**: `BinanceTestnetTradingEngine` (and `BinanceTestnetExecutionEngine` alias/class) in `feature engineering scripts/inference/binance_testnet_engine.py`, integrating `maker_node.py` primitives (`QueuePriorityTracker`, `AdverseSelectionGuard`, `BBOExecutionNode`, contracts `MakerOrderIntent`, `ExecutionSimulationEvent`). Support pure Post-Only GTX limit orders, zero taker fallback on -5022, FIFO queue depletion tracking, adverse selection cancellation on depth drop / microprice drift, dynamic ATR barriers, strict 30s horizon, and continuous JSONL ledger streaming. Update `rigorous_evaluation_audit.py` to support `live_streaming_eval_ledger.jsonl` with fallback to historical ledgers.
- **Success criteria**:
  - `pytest tests/test_leakage_and_features.py -v` passes 100%.
  - `python "feature engineering scripts/scratch/rigorous_evaluation_audit.py"` runs without error, calculating Wilson CIs, exact binomial p-values on clustered runs, non-tie accuracy, drift-adjusted alpha, and ECE.
  - `python scripts/purged_walk_forward_backtest.py --fees-bps 4.0 --embargo-bars 3` passes.
- **Interface contracts**: `PROJECT.md` § Interface Contracts, `tests/contracts.py`.
- **Code layout**: `PROJECT.md` § Code Layout.

## Change Tracker
- **Files modified**:
  - `feature engineering scripts/inference/binance_testnet_engine.py`: Implemented `BinanceTestnetClient` and `BinanceTestnetExecutionEngine` with 6-state FSM (`IDLE`, `PLACING`, `RESTING_IN_QUEUE`, `PARTIAL_FILL`, `FILLED_HOLDING`, `CANCELING`), Post-Only GTX limit order placement at BBO, zero taker fallback on -5022, FIFO queue priority tracking, active adverse selection cancellation, dynamic ATR barriers, 30s horizon auto-flattening, and continuous JSONL telemetry streaming.
  - `feature engineering scripts/scratch/rigorous_evaluation_audit.py`: Enhanced to read `scratch/live_streaming_eval_ledger.jsonl` with seamless fallback to historical ledgers, supports both nested and flat schemas, computes Wilson 95% CIs, exact binomial p-values on clustered independent runs, drift-adjusted alpha, and ECE reliability.
- **Build status**: Verification tests passing.
- **Pending issues**: Complete final verification commands and produce handoff.md.

## Quality Status
- **Build/test result**: pytest test_leakage_and_features.py (48/48 PASSED), rigorous_evaluation_audit.py (PASSED), purged_walk_forward_backtest.py (PASSED).
- **Lint status**: Clean (no syntax errors, standard PEP8 conventions).
- **Tests added/modified**: Verified execution against master test suite.

## Loaded Skills
- None specified in prompt.

## Key Decisions Made
- Architecture follows `testnet_execution_strategy.md` and `adversarial_telemetry_strategy.md`.
- Implemented both `BinanceTestnetExecutionEngine` and `BinanceTestnetTradingEngine` (as alias or subclass) so any caller using either name works seamlessly.
- Zero taker fallback on -5022 strictly aborts and returns to IDLE to preserve maker fee tier (4.0 bps RT vs 10.0 bps taker).
- Dynamic ATR barriers set to $1.5 \times \text{ATR}_{10s}$ for TP and $1.2 \times \text{ATR}_{10s}$ for SL with 30s hard time-stop.
- `rigorous_evaluation_audit.py` accepts `--ledger <path>` or defaults to `live_streaming_eval_ledger.jsonl` (with fallback to `live_45m_eval_ledger.jsonl`).

## Artifact Index
- `feature engineering scripts/inference/binance_testnet_engine.py` — Autonomous Binance USD-M testnet execution engine.
- `feature engineering scripts/scratch/rigorous_evaluation_audit.py` — Rigorous audit evaluation engine supporting streaming and historical ledgers.
- `D:\backup\Binance data\.agents\teamwork\worker_testnet_engine\handoff.md` — Final handoff report.

