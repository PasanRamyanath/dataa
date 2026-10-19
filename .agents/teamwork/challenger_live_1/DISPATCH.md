# Dispatch Assignment — challenger_live_1

## Identity
- Archetype: teamwork_preview_challenger
- Role: Streaming Pipeline Adversarial Challenger
- Working Directory: D:\backup\Binance data\.agents\teamwork\challenger_live_1

## Authoritative Inputs
- ORIGINAL_REQUEST.md path: D:\backup\Binance data\.agents\teamwork\ORIGINAL_REQUEST.md
- PROJECT.md path: D:\backup\Binance data\PROJECT.md
- TEST_READY.md path: D:\backup\Binance data\TEST_READY.md
- Worker Handoffs:
  - `D:\backup\Binance data\.agents\teamwork\worker_live_pipeline\handoff.md`

## Challenge Scope & Instructions
1. Read `D:\backup\Binance data\.agents\teamwork\ORIGINAL_REQUEST.md` (mandatory).
2. Empirically challenge the streaming pipeline (`live_loop_v2.py`, `binance_ws.py`, `pipeline.py`):
   - Design stress tests for timestamp cutoff boundaries ($t \le T_{\text{close}} - 1000\text{ ms}$).
   - Test for lookahead or intra-bar leakage under out-of-order ticks, burst arrivals, and socket disconnects.
   - Empirically measure end-to-end turnaround latency under high event frequency to verify compliance with $\le 250\text{ ms}$.
   - Verify synthetic flat-bar interpolation when WebSocket disconnects $> 10\text{s}$.
3. Execute your empirical stress harness and analyze outputs.
4. Issue an unambiguous empirical confirmation: `CONFIRMED` or `DISPROVEN` in `handoff.md`. Send a message to orchestrator with your verdict.


## 2026-10-18T02:53:25Z
From: 6f56ce3c-0c13-49cc-a255-279a76a73637
You are challenger_live_1.
Working Directory: D:\backup\Binance data\.agents\teamwork\challenger_live_1
Read D:\backup\Binance data\.agents\teamwork\ORIGINAL_REQUEST.md (mandatory).
Read your dispatch file at D:\backup\Binance data\.agents\teamwork\challenger_live_1\DISPATCH.md.
Empirically stress-test the live streaming pipeline (cutoff boundaries, zero lookahead leakage, burst latency <= 250 ms, flat-bar interpolation).
Write your handoff report to D:\backup\Binance data\.agents\teamwork\challenger_live_1\handoff.md with explicit confirmation (CONFIRMED or DISPROVEN). Send a message back to the orchestrator.
