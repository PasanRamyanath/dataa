# Dispatch Assignment — reviewer_live_1

## Identity
- Archetype: teamwork_preview_reviewer
- Role: Code Correctness & Interface Conformance Reviewer
- Working Directory: D:\backup\Binance data\.agents\teamwork\reviewer_live_1

## Authoritative Inputs
- ORIGINAL_REQUEST.md path: D:\backup\Binance data\.agents\teamwork\ORIGINAL_REQUEST.md
- PROJECT.md path: D:\backup\Binance data\PROJECT.md
- TEST_READY.md path: D:\backup\Binance data\TEST_READY.md
- Worker Handoffs:
  - `D:\backup\Binance data\.agents\teamwork\worker_live_pipeline\handoff.md`
  - `D:\backup\Binance data\.agents\teamwork\worker_testnet_engine\handoff.md`

## Review Scope & Instructions
1. Read `D:\backup\Binance data\.agents\teamwork\ORIGINAL_REQUEST.md` (mandatory).
2. Review the code changes made by workers in:
   - `feature engineering scripts/streaming/live_ingestion/binance_ws.py`
   - `feature engineering scripts/execution/maker_node.py`
   - `feature engineering scripts/streaming/live_loop_v2.py`
   - `feature engineering scripts/inference/binance_testnet_engine.py`
   - `feature engineering scripts/scratch/rigorous_evaluation_audit.py`
3. Verify interface conformance with contracts in `tests/contracts.py` and `PROJECT.md`.
4. Verify anti-leakage invariants: $t \le T_{\text{close}} - 1000\text{ ms}$, zero lookahead, zero unclosed bar return contamination.
5. Independently run the verification suite:
   - `pytest tests/test_leakage_and_features.py -v`
   - `python scripts/evaluate_ensemble_and_meta.py --data-dir "D:/backup/Binance data"`
6. Document findings and issue an unambiguous verdict (`APPROVE` or `REQUEST_CHANGES`) in `handoff.md`. Send a message to orchestrator with your verdict.

## 2026-10-18T02:53:25Z
You are reviewer_live_1.
Working Directory: D:\backup\Binance data\.agents\teamwork\reviewer_live_1
Read D:\backup\Binance data\.agents\teamwork\ORIGINAL_REQUEST.md (mandatory).
Read your dispatch file at D:\backup\Binance data\.agents\teamwork\reviewer_live_1\DISPATCH.md.
Review code changes for correctness, interface conformance, and anti-leakage invariants.
Run verification commands: pytest tests/test_leakage_and_features.py -v and python scripts/evaluate_ensemble_and_meta.py --data-dir "D:/backup/Binance data".
Write your handoff report to D:\backup\Binance data\.agents\teamwork\reviewer_live_1\handoff.md with an explicit verdict (APPROVE or REQUEST_CHANGES). Send a message back to the orchestrator.
