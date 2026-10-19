# Dispatch Assignment — reviewer_live_2

## Identity
- Archetype: teamwork_preview_reviewer
- Role: Robustness & Testnet Execution Reviewer
- Working Directory: D:\backup\Binance data\.agents\teamwork\reviewer_live_2

## Authoritative Inputs
- ORIGINAL_REQUEST.md path: D:\backup\Binance data\.agents\teamwork\ORIGINAL_REQUEST.md
- PROJECT.md path: D:\backup\Binance data\PROJECT.md
- TEST_READY.md path: D:\backup\Binance data\TEST_READY.md
- Worker Handoffs:
  - `D:\backup\Binance data\.agents\teamwork\worker_live_pipeline\handoff.md`
  - `D:\backup\Binance data\.agents\teamwork\worker_testnet_engine\handoff.md`

## Review Scope & Instructions
1. Read `D:\backup\Binance data\.agents\teamwork\ORIGINAL_REQUEST.md` (mandatory).
2. Review the robustness and execution mechanics:
   - Live testnet engine (`binance_testnet_engine.py`): pure GTX limit orders, zero taker fallback on -5022, live FIFO queue tracking, active adverse selection cancel, dynamic ATR barriers, 30s horizon auto-flattening.
   - Live streaming latency: confirm $\le 250\text{ ms}$ budget compliance.
   - Continuous telemetry ledger: confirm logging schema and forward mid-price return resolution in `scratch/live_streaming_eval_ledger.jsonl`.
3. Independently run the verification suite:
   - `pytest tests/test_leakage_and_features.py -v`
   - `python "feature engineering scripts/scratch/rigorous_evaluation_audit.py"`
   - `python scripts/purged_walk_forward_backtest.py --fees-bps 4.0 --embargo-bars 3`
4. Document findings and issue an unambiguous verdict (`APPROVE` or `REQUEST_CHANGES`) in `handoff.md`. Send a message to orchestrator with your verdict.


## 2026-10-18T02:53:25Z
From: 6f56ce3c-0c13-49cc-a255-279a76a73637
Priority: MESSAGE_PRIORITY_HIGH
Content:
You are reviewer_live_2.
Working Directory: D:\backup\Binance data\.agents\teamwork\reviewer_live_2
Read D:\backup\Binance data\.agents\teamwork\ORIGINAL_REQUEST.md (mandatory).
Read your dispatch file at D:\backup\Binance data\.agents\teamwork\reviewer_live_2\DISPATCH.md.
Review robustness, latency budget (<= 250 ms), testnet execution mechanics, and telemetry logging schema.
Run verification commands: pytest tests/test_leakage_and_features.py -v, python "feature engineering scripts/scratch/rigorous_evaluation_audit.py", and python scripts/purged_walk_forward_backtest.py --fees-bps 4.0 --embargo-bars 3.
Write your handoff report to D:\backup\Binance data\.agents\teamwork\reviewer_live_2\handoff.md with an explicit verdict (APPROVE or REQUEST_CHANGES). Send a message back to the orchestrator.
