# Dispatch Assignment — explorer_r1_r4_audit

## Identity
- Archetype: teamwork_preview_explorer
- Role: Adversarial Audit & Telemetry Specialist
- Working Directory: D:\backup\Binance data\.agents\teamwork\explorer_r1_r4_audit

## Authoritative Inputs
- ORIGINAL_REQUEST.md path: D:\backup\Binance data\.agents\teamwork\ORIGINAL_REQUEST.md
- PROJECT.md path: D:\backup\Binance data\PROJECT.md
- TEST_READY.md path: D:\backup\Binance data\TEST_READY.md
- Target files:
  - `feature engineering scripts/scratch/rigorous_evaluation_audit.py`
  - `feature engineering scripts/scratch/adversarial_leakage_verifier.py`
  - `scripts/purged_walk_forward_backtest.py`
  - `scripts/evaluate_ensemble_and_meta.py`
  - `scratch/live_streaming_eval_ledger.jsonl`

## Objective & Instructions
1. Read `D:\backup\Binance data\.agents\teamwork\ORIGINAL_REQUEST.md` (mandatory).
2. Investigate adversarial stress testing and edge cases across `features/`, `meta/`, and `execution/`: zero liquidity, inverted basis, flat/zero volatility chop, missing bookTicker ticks, infinite queue depth.
3. Investigate the automated continuous telemetry ledger format in `scratch/live_streaming_eval_ledger.jsonl`: fields for live predictions, BBO quotes, queue state, order intent, fill outcome, and forward return resolution ($H1, H2, H3, H6, H12, H18$).
4. Examine how `rigorous_evaluation_audit.py` evaluates the telemetry ledger (Wilson 95% CIs, exact binomial p-values, non-tie accuracy, drift-adjusted alpha).
5. Write your complete findings and implementation plan to `D:\backup\Binance data\.agents\teamwork\explorer_r1_r4_audit\adversarial_telemetry_strategy.md`.
6. Write `handoff.md` and send a concise message back to the orchestrator.

## 2026-10-18T02:29:05Z
You are explorer_r1_r4_audit.
Working Directory: D:\backup\Binance data\.agents\teamwork\explorer_r1_r4_audit
Read D:\backup\Binance data\.agents\teamwork\ORIGINAL_REQUEST.md (mandatory).
Read your dispatch file at D:\backup\Binance data\.agents\teamwork\explorer_r1_r4_audit\DISPATCH.md.
Investigate adversarial stress testing (features/, meta/, execution/ edge cases: zero liquidity, inverted basis, zero volatility chop, missing ticks) and continuous telemetry ledger logging (scratch/live_streaming_eval_ledger.jsonl and rigorous_evaluation_audit.py). Formulate the technical strategy to achieve 100% robustness and automated forward return evaluation.
Write your report to D:\backup\Binance data\.agents\teamwork\explorer_r1_r4_audit\adversarial_telemetry_strategy.md and your handoff report to D:\backup\Binance data\.agents\teamwork\explorer_r1_r4_audit\handoff.md.
Send a message back to the orchestrator with your findings.

