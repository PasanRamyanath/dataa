# Dispatch Assignment — worker_live_pipeline

## Identity
- Archetype: teamwork_preview_worker
- Role: Live Ingestion & Streaming Loop Specialist
- Working Directory: D:\backup\Binance data\.agents\teamwork\worker_live_pipeline

## Authoritative Inputs
- ORIGINAL_REQUEST.md path: D:\backup\Binance data\.agents\teamwork\ORIGINAL_REQUEST.md
- PROJECT.md path: D:\backup\Binance data\PROJECT.md
- TEST_READY.md path: D:\backup\Binance data\TEST_READY.md
- Explorer Strategy Reports:
  - `D:\backup\Binance data\.agents\teamwork\explorer_r2_streaming\streaming_integration_strategy.md`
  - `D:\backup\Binance data\.agents\teamwork\explorer_r1_r4_audit\adversarial_telemetry_strategy.md`

## Exclusive Write Ownership
You exclusively own and may modify:
- `feature engineering scripts/streaming/live_ingestion/binance_ws.py`
- `feature engineering scripts/execution/maker_node.py`
- `feature engineering scripts/streaming/live_loop_v2.py`
You MUST NOT modify files owned by other agents (`binance_testnet_engine.py`, `tests/`, etc.).

## Objectives & Detailed Tasks
1. Read `D:\backup\Binance data\.agents\teamwork\ORIGINAL_REQUEST.md` (mandatory).
2. Read the strategies in `streaming_integration_strategy.md` and `adversarial_telemetry_strategy.md`.
3. In `binance_ws.py`:
   - Add `ethusdt@aggTrade` and `ethusdt@bookTicker` streams to `BinanceLiveIngestor` to feed Cross-Asset Family 1.
   - Enforce strict $1000\text{ ms}$ feature cutoff ($t \le T_{\text{close}} - 1000\text{ ms}$) with execution freeze partition ($T_{\text{close}} - 1000\text{ ms} < t \le T_{\text{close}}$).
   - Add synthetic flat bar interpolation if socket timeout/drop exceeds $10\text{s}$ to prevent ring buffer discontinuities.
4. In `maker_node.py`:
   - Implement pre-trade quote viability check in `BBOExecutionNode` (`bbo_bid > 0`, `bbo_ask > bbo_bid`, `bbo_bid_qty > 0`, `bbo_ask_qty > 0`) to prevent unhandled crashes on zero liquidity.
   - Implement crossed-book spread guard (`bbo_ask - bbo_bid >= 0.10 USDT`).
   - Distinguish fee accounting for un-filled adverse cancellations ($0.0\text{ bps}$) versus filled trades ($4.0\text{ bps}$).
5. In `live_loop_v2.py`:
   - Wire `compute_orthogonal_features` using the sanitized feeds.
   - Wire `load_orthogonal_ensemble()` and `predict_orthogonal_ensemble()`.
   - Replace legacy `_fuse()` threshold clamping (`p_sel < 0.62`, `0.42` fallbacks) with `ContinuousMetaLabeler`.
   - Wire `BBOExecutionNode` for order intent generation, dynamic ATR barriers, and 30s horizon.
   - Implement continuous streaming telemetry output to `scratch/live_streaming_eval_ledger.jsonl`.
   - Maintain inference latency $\le 250\text{ ms}$ (expected ~16–26 ms).
6. Verify your implementation by running:
   - `pytest tests/test_leakage_and_features.py -v`
   - `python scripts/evaluate_ensemble_and_meta.py --data-dir "D:/backup/Binance data"`
7. Write your handoff report to `D:\backup\Binance data\.agents\teamwork\worker_live_pipeline\handoff.md`.


## 2026-10-18T02:37:23Z
[Message] timestamp=2026-10-18T02:37:23Z sender=6f56ce3c-0c13-49cc-a255-279a76a73637 priority=MESSAGE_PRIORITY_HIGH content=You are worker_live_pipeline.
Working Directory: D:\backup\Binance data\.agents\teamwork\worker_live_pipeline
Read D:\backup\Binance data\.agents\teamwork\ORIGINAL_REQUEST.md (mandatory).
Read your dispatch file at D:\backup\Binance data\.agents\teamwork\worker_live_pipeline\DISPATCH.md.
Follow the implementation strategies in:
- D:\backup\Binance data\.agents\teamwork\explorer_r2_streaming\streaming_integration_strategy.md
- D:\backup\Binance data\.agents\teamwork\explorer_r1_r4_audit\adversarial_telemetry_strategy.md
Implement the streaming ingestion hardening (binance_ws.py), defensive guards (maker_node.py), and live streaming loop overhaul (live_loop_v2.py) with continuous telemetry logging to scratch/live_streaming_eval_ledger.jsonl.
Run verification tests (pytest tests/test_leakage_and_features.py -v and python scripts/evaluate_ensemble_and_meta.py --data-dir "D:/backup/Binance data").
Write your handoff report to D:\backup\Binance data\.agents\teamwork\worker_live_pipeline\handoff.md.

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

