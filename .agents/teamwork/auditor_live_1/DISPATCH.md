# Dispatch Assignment — auditor_live_1

## Identity
- Archetype: teamwork_preview_auditor
- Role: Forensic Integrity Auditor
- Working Directory: D:\backup\Binance data\.agents\teamwork\auditor_live_1

## Authoritative Inputs
- ORIGINAL_REQUEST.md path: D:\backup\Binance data\.agents\teamwork\ORIGINAL_REQUEST.md
- PROJECT.md path: D:\backup\Binance data\PROJECT.md
- TEST_READY.md path: D:\backup\Binance data\TEST_READY.md
- Target Files & Deliverables:
  - `feature engineering scripts/streaming/live_ingestion/binance_ws.py`
  - `feature engineering scripts/execution/maker_node.py`
  - `feature engineering scripts/streaming/live_loop_v2.py`
  - `feature engineering scripts/inference/binance_testnet_engine.py`
  - `feature engineering scripts/scratch/rigorous_evaluation_audit.py`
  - `scratch/live_streaming_eval_ledger.jsonl`

## Audit Scope & Instructions
1. Read `D:\backup\Binance data\.agents\teamwork\ORIGINAL_REQUEST.md` (mandatory).
2. Perform comprehensive Forensic Integrity Verification:
   - Check git diff across all modified and untracked files for hardcoded outputs, dummy/mock logic, or test bypasses.
   - Verify that all calculations (3-family orthogonal ensemble, continuous Platt scaling, FIFO queue tracking, adverse selection cancellation, dynamic ATR barriers, Wilson score CIs) are genuine, mathematically sound, and executed faithfully.
   - Verify zero leakage: ensure no future timestamps, unclosed return references, or lookahead shortcuts exist.
   - Independently execute all 4 verification commands:
     - `pytest tests/test_leakage_and_features.py -v`
     - `python scripts/evaluate_ensemble_and_meta.py --data-dir "D:/backup/Binance data"`
     - `python scripts/purged_walk_forward_backtest.py --fees-bps 4.0 --embargo-bars 3`
     - `python "feature engineering scripts/scratch/rigorous_evaluation_audit.py"`
3. Issue an unambiguous binary verdict in `handoff.md`:
   - `CLEAN` (zero cheating, genuine implementation, all tests pass)
   - OR `INTEGRITY VIOLATION` (with detailed evidence).
4. Send a message to orchestrator with your verdict.


## 2026-10-18T02:53:25Z
You are auditor_live_1.
Working Directory: D:\backup\Binance data\.agents\teamwork\auditor_live_1
Read D:\backup\Binance data\.agents\teamwork\ORIGINAL_REQUEST.md (mandatory).
Read your dispatch file at D:\backup\Binance data\.agents\teamwork\auditor_live_1\DISPATCH.md.
Perform forensic integrity verification: check git diff for fake/hardcoded outputs, facades, or test circumvention; verify genuine mathematical calculations; independently run all 4 verification commands.
Write your handoff report to D:\backup\Binance data\.agents\teamwork\auditor_live_1\handoff.md with explicit verdict (CLEAN or INTEGRITY VIOLATION). Send a message back to the orchestrator.
