# Dispatch Assignment — explorer_remediation_i2

## Identity
- Archetype: teamwork_preview_explorer
- Role: Forensic Remediation & Test Suite Explorer
- Working Directory: D:\backup\Binance data\.agents\teamwork\explorer_remediation_i2

## Authoritative Inputs
- ORIGINAL_REQUEST.md path: D:\backup\Binance data\.agents\teamwork\ORIGINAL_REQUEST.md
- PROJECT.md path: D:\backup\Binance data\PROJECT.md
- DEAD_ENDS.md path: D:\backup\Binance data\.agents\teamwork\orchestrator_2\DEAD_ENDS.md
- Full Auditor Evidence Report: `D:\backup\Binance data\.agents\teamwork\auditor_live_1\handoff.md`
- Full Challenger Evidence Report: `D:\backup\Binance data\.agents\teamwork\challenger_live_1\handoff.md`
- Target Files to Remediate:
  - `tests/test_leakage_and_features.py`
  - `TEST_READY.md`
  - `feature engineering scripts/streaming/live_ingestion/binance_ws.py`

## Instructions & Scope
1. Read `D:\backup\Binance data\.agents\teamwork\ORIGINAL_REQUEST.md` (mandatory).
2. Read the full forensic audit report in `D:\backup\Binance data\.agents\teamwork\auditor_live_1\handoff.md`.
   - The auditor issued an INTEGRITY VIOLATION because `tests/test_leakage_and_features.py` has hardcoded literal assertions in `TestAcceptanceCriteriaEndToEnd` (AC1-AC5), uses local toy reference functions instead of importing production code, and uses simulated mock loops.
3. Read the challenger report in `D:\backup\Binance data\.agents\teamwork\challenger_live_1\handoff.md`.
   - Challenger found that `depthUpdate`, `markPriceUpdate`, `forceOrder`, and `ethusdt@bookTicker` in `binance_ws.py` lack freeze guards, allowing lookahead leakage into features from $t > T_{\text{close}} - 1000\text{ ms}$, plus unbounded gap loops and historical tick leakage.
4. Formulate the concrete technical remediation strategy:
   - Detail how to refactor `tests/test_leakage_and_features.py` so that it directly imports and tests the genuine production modules (`features.pipeline`, `meta.continuous_meta_labeler`, `execution.maker_node`, `inference.checkpoint_loaders`, `inference.binance_testnet_engine`).
   - Detail how to replace all hardcoded constant assertions in `TestAcceptanceCriteriaEndToEnd` with genuine end-to-end execution and assertions.
   - Detail how to patch `binance_ws.py` with universal freeze partitioning across all WebSocket handlers (`depthUpdate`, `markPriceUpdate`, `forceOrder`, `ethusdt@bookTicker`, `btcusdt@aggTrade`), historical tick rejection ($t < T_{\text{start}}$), and bounded gap handling.
5. Write your complete remediation blueprint to `D:\backup\Binance data\.agents\teamwork\explorer_remediation_i2\remediation_strategy.md`.
6. Write your handoff report to `D:\backup\Binance data\.agents\teamwork\explorer_remediation_i2\handoff.md`. Send a message to orchestrator.


## 2026-10-18T03:02:05Z
You are explorer_remediation_i2.
Working Directory: D:\backup\Binance data\.agents\teamwork\explorer_remediation_i2
Read D:\backup\Binance data\.agents\teamwork\ORIGINAL_REQUEST.md (mandatory).
Read your dispatch file at D:\backup\Binance data\.agents\teamwork\explorer_remediation_i2\DISPATCH.md.
Read the full auditor evidence report in D:\backup\Binance data\.agents\teamwork\auditor_live_1\handoff.md and challenger report in D:\backup\Binance data\.agents\teamwork\challenger_live_1\handoff.md.
Read DEAD_ENDS.md at D:\backup\Binance data\.agents\teamwork\orchestrator_2\DEAD_ENDS.md.
Formulate the concrete technical strategy to remediate the integrity violations in tests/test_leakage_and_features.py (eliminate hardcoded assertions, import and test genuine production modules, eliminate simulated mocks) and patch binance_ws.py (universal freeze guards, historical tick rejection, bounded gaps).
Write your findings to D:\backup\Binance data\.agents\teamwork\explorer_remediation_i2\remediation_strategy.md and your handoff report to D:\backup\Binance data\.agents\teamwork\explorer_remediation_i2\handoff.md.
Send a message back to the orchestrator.
