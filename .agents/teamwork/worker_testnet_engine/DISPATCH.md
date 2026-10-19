# Dispatch Assignment — worker_testnet_engine

## Identity
- Archetype: teamwork_preview_worker
- Role: Testnet Execution Engine Specialist
- Working Directory: D:\backup\Binance data\.agents\teamwork\worker_testnet_engine

## Authoritative Inputs
- ORIGINAL_REQUEST.md path: D:\backup\Binance data\.agents\teamwork\ORIGINAL_REQUEST.md
- PROJECT.md path: D:\backup\Binance data\PROJECT.md
- TEST_READY.md path: D:\backup\Binance data\TEST_READY.md
- Explorer Strategy Reports:
  - `D:\backup\Binance data\.agents\teamwork\explorer_r3_testnet\testnet_execution_strategy.md`
  - `D:\backup\Binance data\.agents\teamwork\explorer_r1_r4_audit\adversarial_telemetry_strategy.md`

## Exclusive Write Ownership
You exclusively own and may modify:
- `feature engineering scripts/inference/binance_testnet_engine.py`
- `feature engineering scripts/scratch/rigorous_evaluation_audit.py`
You MUST NOT modify files owned by other agents (`binance_ws.py`, `maker_node.py`, `live_loop_v2.py`, `tests/`, etc.).

## Objectives & Detailed Tasks
1. Read `D:\backup\Binance data\.agents\teamwork\ORIGINAL_REQUEST.md` (mandatory).
2. Read the strategies in `testnet_execution_strategy.md` and `adversarial_telemetry_strategy.md`.
3. In `feature engineering scripts/inference/binance_testnet_engine.py`:
   - Implement `BinanceTestnetExecutionEngine` integrating `BBOExecutionNode` from `feature engineering scripts/execution/maker_node.py`.
   - Place pure Post-Only (`GTX`) limit orders at BBO (`timeInForce="GTX"`).
   - Zero taker fallback: on API code -5022 (`Post Only order would immediately match and take`), immediately cancel and wait for the next bar rather than falling back to taker market orders.
   - Implement live FIFO queue tracking: initialize $Q_{\text{ahead}}$ from BBO depth and decrement as incoming `@aggTrade` fills occur at that price level.
   - Implement active adverse selection cancellation: monitor `@depth20` and `@bookTicker`; if queue moves against the order or order book imbalance deteriorates, cancel immediately.
   - Implement dynamic ATR barriers ($1.5 \times \text{ATR}_{10s}$ TP, $1.2 \times \text{ATR}_{10s}$ SL) and strict 30s horizon auto-flattening.
   - Implement 6-state deterministic FSM (`IDLE`, `PLACING`, `RESTING_IN_QUEUE`, `PARTIAL_FILL`, `FILLED_HOLDING`, `CANCELING`).
   - Stream real-time order intents, fills, cancellations, and forward mid returns to `scratch/live_streaming_eval_ledger.jsonl`.
4. In `feature engineering scripts/scratch/rigorous_evaluation_audit.py`:
   - Ensure the evaluation script reads `scratch/live_streaming_eval_ledger.jsonl` (and supports historical fallbacks).
   - Reproduce all metrics: Wilson 95% CIs, exact binomial p-values on clustered independent runs, non-tie accuracy, drift-adjusted alpha, and ECE reliability.
5. Verify your implementation by running:
   - `pytest tests/test_leakage_and_features.py -v`
   - `python "feature engineering scripts/scratch/rigorous_evaluation_audit.py"`
   - `python scripts/purged_walk_forward_backtest.py --fees-bps 4.0 --embargo-bars 3`
6. Write your handoff report to `D:\backup\Binance data\.agents\teamwork\worker_testnet_engine\handoff.md`.

## MANDATORY INTEGRITY WARNING
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

## 2026-10-18T02:37:23Z
You are worker_testnet_engine.
Working Directory: D:\backup\Binance data\.agents\teamwork\worker_testnet_engine
Read D:\backup\Binance data\.agents\teamwork\ORIGINAL_REQUEST.md (mandatory).
Read your dispatch file at D:\backup\Binance data\.agents\teamwork\worker_testnet_engine\DISPATCH.md.
Follow the implementation strategies in:
- D:\backup\Binance data\.agents\teamwork\explorer_r3_testnet\testnet_execution_strategy.md
- D:\backup\Binance data\.agents\teamwork\explorer_r1_r4_audit\adversarial_telemetry_strategy.md
Implement the autonomous Binance USD-M testnet execution engine (binance_testnet_engine.py) integrating BBOExecutionNode (GTX Post-Only, zero taker fallback, FIFO queue tracking, adverse selection cancellation, 30s horizon, dynamic ATR barriers, telemetry logging to scratch/live_streaming_eval_ledger.jsonl) and rigorous_evaluation_audit.py.
Run verification tests (pytest tests/test_leakage_and_features.py -v, python "feature engineering scripts/scratch/rigorous_evaluation_audit.py", and python scripts/purged_walk_forward_backtest.py --fees-bps 4.0 --embargo-bars 3).
Write your handoff report to D:\backup\Binance data\.agents\teamwork\worker_testnet_engine\handoff.md.

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.
