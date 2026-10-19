# Dispatch Assignment — explorer_r3_testnet

## Identity
- Archetype: teamwork_preview_explorer
- Role: Testnet Execution Engine Specialist
- Working Directory: D:\backup\Binance data\.agents\teamwork\explorer_r3_testnet

## Authoritative Inputs
- ORIGINAL_REQUEST.md path: D:\backup\Binance data\.agents\teamwork\ORIGINAL_REQUEST.md
- PROJECT.md path: D:\backup\Binance data\PROJECT.md
- TEST_READY.md path: D:\backup\Binance data\TEST_READY.md
- Target files:
  - `feature engineering scripts/execution/maker_node.py`
  - `feature engineering scripts/inference/binance_testnet_engine.py` (check existence or create/integrate)
  - `feature engineering scripts/scratch/test_binance_testnet_*.py`
  - `tests/contracts.py`

## Objective & Instructions
1. Read `D:\backup\Binance data\.agents\teamwork\ORIGINAL_REQUEST.md` (mandatory).
2. Investigate the execution architecture: `BBOExecutionNode`, `QueuePriorityTracker`, and `AdverseSelectionGuard` in `feature engineering scripts/execution/maker_node.py`.
3. Investigate the testnet integration requirements (Binance USD-M testnet API endpoints, authentication, placing `GTX` Post-Only limit orders at BBO, cancellation on book deterioration, 30s holding horizon, dynamic ATR barriers).
4. Determine how `binance_testnet_engine.py` should be implemented or integrated with `BBOExecutionNode` for autonomous live testnet execution.
5. Write your complete findings and implementation plan to `D:\backup\Binance data\.agents\teamwork\explorer_r3_testnet\testnet_execution_strategy.md`.
6. Write `handoff.md` and send a concise message back to the orchestrator.


## 2026-10-18T02:29:05Z
From: 6f56ce3c-0c13-49cc-a255-279a76a73637
You are explorer_r3_testnet.
Working Directory: D:\backup\Binance data\.agents\teamwork\explorer_r3_testnet
Read D:\backup\Binance data\.agents\teamwork\ORIGINAL_REQUEST.md (mandatory).
Read your dispatch file at D:\backup\Binance data\.agents\teamwork\explorer_r3_testnet\DISPATCH.md.
Investigate BBOExecutionNode in maker_node.py and binance_testnet_engine.py. Formulate the technical strategy to integrate autonomous BBO maker execution on Binance USD-M testnet (GTX limit orders, FIFO queue tracking, adverse selection cancellation, 30s horizon, dynamic ATR barriers).
Write your report to D:\backup\Binance data\.agents\teamwork\explorer_r3_testnet\testnet_execution_strategy.md and your handoff report to D:\backup\Binance data\.agents\teamwork\explorer_r3_testnet\handoff.md.
Send a message back to the orchestrator with your findings.
