# Dispatch Assignment — explorer_r2_streaming

## Identity
- Archetype: teamwork_preview_explorer
- Role: Streaming Ingestion & Live Loop Specialist
- Working Directory: D:\backup\Binance data\.agents\teamwork\explorer_r2_streaming

## Authoritative Inputs
- ORIGINAL_REQUEST.md path: D:\backup\Binance data\.agents\teamwork\ORIGINAL_REQUEST.md
- PROJECT.md path: D:\backup\Binance data\PROJECT.md
- TEST_READY.md path: D:\backup\Binance data\TEST_READY.md
- Target files:
  - `feature engineering scripts/streaming/live_loop_v2.py`
  - `feature engineering scripts/streaming/live_ingestion/binance_ws.py`
  - `feature engineering scripts/features/pipeline.py`
  - `feature engineering scripts/meta/continuous_meta_labeler.py`

## Objective & Instructions
1. Read `D:\backup\Binance data\.agents\teamwork\ORIGINAL_REQUEST.md` (mandatory).
2. Investigate how `live_loop_v2.py` and `binance_ws.py` currently ingest WebSocket streams (bookTicker, aggTrade) and how models are called.
3. Determine exact steps to wire the decoupled orthogonal feature pipeline (`compute_orthogonal_features`) and continuous calibrated meta-labeler (`ContinuousMetaLabeler`) into `live_loop_v2.py`.
4. Ensure strict timestamp cutoff: $t_{event} \le T_{close} - 1000\text{ ms}$, zero lookahead leakage, and inference latency $\le 250\text{ ms}$.
5. Write your complete findings and implementation plan to `D:\backup\Binance data\.agents\teamwork\explorer_r2_streaming\streaming_integration_strategy.md`.
6. Write `handoff.md` and send a concise message back to the orchestrator.


## 2026-10-18T02:29:05Z
You are explorer_r2_streaming.
Working Directory: D:\backup\Binance data\.agents\teamwork\explorer_r2_streaming
Read D:\backup\Binance data\.agents\teamwork\ORIGINAL_REQUEST.md (mandatory).
Read your dispatch file at D:\backup\Binance data\.agents\teamwork\explorer_r2_streaming\DISPATCH.md.
Investigate live_loop_v2.py and binance_ws.py. Formulate the technical strategy to wire orthogonal features and continuous meta-labeler into the live streaming loop while maintaining strict t <= T_close - 1000ms timestamp offset and latency <= 250ms.
Write your report to D:\backup\Binance data\.agents\teamwork\explorer_r2_streaming\streaming_integration_strategy.md and your handoff report to D:\backup\Binance data\.agents\teamwork\explorer_r2_streaming\handoff.md.
Send a message back to the orchestrator with your findings.
