# Dispatch Assignment — worker_remediation_i2

## Identity
- Archetype: teamwork_preview_worker
- Role: Forensic Remediation & Test Suite Overhaul Worker
- Working Directory: D:\backup\Binance data\.agents\teamwork\worker_remediation_i2

## Authoritative Inputs
- ORIGINAL_REQUEST.md path: D:\backup\Binance data\.agents\teamwork\ORIGINAL_REQUEST.md
- PROJECT.md path: D:\backup\Binance data\PROJECT.md
- DEAD_ENDS.md path: D:\backup\Binance data\.agents\teamwork\orchestrator_2\DEAD_ENDS.md
- Remediation Strategy: `D:\backup\Binance data\.agents\teamwork\explorer_remediation_i2\remediation_strategy.md`
- Remediation Handoff: `D:\backup\Binance data\.agents\teamwork\explorer_remediation_i2\handoff.md`

## Exclusive Write Ownership
You exclusively own and may modify:
- `tests/test_leakage_and_features.py`
- `TEST_READY.md`
- `feature engineering scripts/streaming/live_ingestion/binance_ws.py`
You MUST NOT modify files owned by other modules.

## Objectives & Tasks
1. Read `D:\backup\Binance data\.agents\teamwork\ORIGINAL_REQUEST.md` (mandatory).
2. Follow the detailed blueprints in `remediation_strategy.md`:
   - In `feature engineering scripts/streaming/live_ingestion/binance_ws.py`:
     - Apply universal freeze partitioning across ALL WebSocket handlers (`depthUpdate`, `markPriceUpdate`, `forceOrder`, `ethusdt@bookTicker`, and `btcusdt@aggTrade`) so that ANY event with $t > T_{\text{close}} - 1000\text{ ms}$ is quarantined to the freeze buffer and NOT included in the current bar's feature calculation.
     - Add historical tick rejection (`if event_ts < self.current_bar_start: return`).
     - Add upper bound clamping to `_handle_boundary_gap` (`gap_count = min(raw_gap, 180)`).
   - In `tests/test_leakage_and_features.py`:
     - Completely eliminate the toy reference function `reference_compute_orthogonal_features`.
     - Directly import and test the genuine production modules (`features.pipeline`, `meta.continuous_meta_labeler`, `execution.maker_node`, `inference.checkpoint_loaders`, `inference.binance_testnet_engine`).
     - Eliminate ALL hardcoded literal assertions in `TestAcceptanceCriteriaEndToEnd` (AC1 to AC5); execute real mathematical evaluations against production models and assert against genuine outputs.
     - Eliminate mock loops in `test_model_inference_latency_budget_ac5` (benchmark real model inference latency) and synthetic Gaussian sampling in `test_purged_walk_forward_backtest_net_pnl_ac4` (run genuine walk-forward simulation).
   - In `TEST_READY.md`:
     - Update to accurately reflect true end-to-end integration testing against genuine production modules.
3. Verify your implementation by running:
   - `pytest tests/test_leakage_and_features.py -v`
   - `python "tests/test_feature_leakage_impact.py"`
   - `python scripts/evaluate_ensemble_and_meta.py --data-dir "D:/backup/Binance data"`
   - `python scripts/purged_walk_forward_backtest.py --fees-bps 4.0 --embargo-bars 3`
   - `python "feature engineering scripts/scratch/rigorous_evaluation_audit.py"`
4. Write your handoff report to `D:\backup\Binance data\.agents\teamwork\worker_remediation_i2\handoff.md`.

## MANDATORY INTEGRITY WARNING
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.


## 2026-10-18T03:09:00Z
Received dispatch message:
You are worker_remediation_i2.
Working Directory: D:\backup\Binance data\.agents\teamwork\worker_remediation_i2
Read D:\backup\Binance data\.agents\teamwork\ORIGINAL_REQUEST.md (mandatory).
Read your dispatch file at D:\backup\Binance data\.agents\teamwork\worker_remediation_i2\DISPATCH.md.
Follow the implementation blueprint in D:\backup\Binance data\.agents\teamwork\explorer_remediation_i2\remediation_strategy.md.
Patch binance_ws.py with universal freeze guards, historical tick rejection, and bounded gap handling.
Refactor tests/test_leakage_and_features.py to eliminate toy reference functions, import genuine production modules, eliminate hardcoded constant assertions, and run real algorithmic tests. Update TEST_READY.md.
Run verification tests (pytest tests/test_leakage_and_features.py -v, python tests/test_feature_leakage_impact.py, evaluate_ensemble_and_meta.py, purged_walk_forward_backtest.py, and rigorous_evaluation_audit.py).
Write your handoff report to D:\backup\Binance data\.agents\teamwork\worker_remediation_i2\handoff.md.

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.
