# Dispatch Assignment — challenger_live_2

## Identity
- Archetype: teamwork_preview_challenger
- Role: Testnet Execution Adversarial Challenger
- Working Directory: D:\backup\Binance data\.agents\teamwork\challenger_live_2

## Authoritative Inputs
- ORIGINAL_REQUEST.md path: D:\backup\Binance data\.agents\teamwork\ORIGINAL_REQUEST.md
- PROJECT.md path: D:\backup\Binance data\PROJECT.md
- TEST_READY.md path: D:\backup\Binance data\TEST_READY.md
- Worker Handoffs:
  - `D:\backup\Binance data\.agents\teamwork\worker_testnet_engine\handoff.md`

## Challenge Scope & Instructions
1. Read `D:\backup\Binance data\.agents\teamwork\ORIGINAL_REQUEST.md` (mandatory).
2. Empirically challenge the testnet execution engine (`binance_testnet_engine.py` and `maker_node.py`):
   - Design stress tests for pure Post-Only GTX limit orders: verify that on API code -5022, the engine NEVER falls back to taker market orders.
   - Empirically verify FIFO queue decrementation against synthetic incoming aggressive volume.
   - Empirically verify adverse selection cancellation triggers on book collapse ($\ge 50\%$) or microprice drift ($> 2.0\text{ bps}$).
   - Empirically verify dynamic ATR barriers ($1.5 \times \text{ATR}_{10s}$ TP, $1.2 \times \text{ATR}_{10s}$ SL) and strict 30s horizon auto-flattening.
   - Verify statistical rigor of Wilson 95% CIs and binomial p-values on clustered runs.
3. Execute your empirical stress harness and analyze outputs.
4. Issue an unambiguous empirical confirmation: `CONFIRMED` or `DISPROVEN` in `handoff.md`. Send a message to orchestrator with your verdict.


## 2026-10-18T02:53:25Z
You are challenger_live_2.
Working Directory: D:\backup\Binance data\.agents\teamwork\challenger_live_2
Read D:\backup\Binance data\.agents\teamwork\ORIGINAL_REQUEST.md (mandatory).
Read your dispatch file at D:\backup\Binance data\.agents\teamwork\challenger_live_2\DISPATCH.md.
Empirically stress-test the testnet maker execution engine (pure GTX, zero taker fallback on -5022, FIFO queue tracking decrementation, adverse selection cancellation, dynamic ATR barriers, 30s horizon auto-flattening, and statistical metrics).
Write your handoff report to D:\backup\Binance data\.agents\teamwork\challenger_live_2\handoff.md with explicit confirmation (CONFIRMED or DISPROVEN). Send a message back to the orchestrator.
