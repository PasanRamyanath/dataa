## 2026-10-18T01:21:15Z
[Message] timestamp=2026-10-18T01:21:15Z sender=4a8e5400-290d-44ec-9df4-e3142d1475c4 priority=MESSAGE_PRIORITY_HIGH content=You are the Spec & Ledger Miner for the BTCUSDT 10-Second Alpha & Execution Overhaul project.

Your Identity & Working Directory:
- Working Directory: D:\backup\Binance data\.agents\teamwork\spec_miner_survey
- Parent Conversation ID: 4a8e5400-290d-44ec-9df4-e3142d1475c4

MANDATORY FIRST STEP:
Read D:\backup\Binance data\.agents\teamwork\ORIGINAL_REQUEST.md completely.

Your Mission:
Conduct an exhaustive specification and empirical ledger mining survey:
1. Deeply analyze the empirical audit findings and raw data in `scratch/live_45m_eval_ledger.jsonl` and `feature engineering scripts/scratch/rigorous_evaluation_audit.py` (and any related audit/ledger files in the workspace).
2. Document the exact formulas, statistical definitions, and threshold requirements for:
   - Directional accuracy (raw vs non-tie vs latency-adjusted)
   - Clustered runs, exact binomial p-values vs H0: p=0.50, Wilson 95% score confidence intervals
   - Alpha decay curve (20-30s / 3 bars, mean bps move, holding horizon)
   - Brier score baseline (climatological prior 0.2786) and Expected Calibration Error (ECE <= 8%)
   - Round-trip maker fee cost (Binance USD-M VIP0: 2 bps maker = 4 bps round-trip) vs taker (10 bps)
   - Volatility hurdles, ATR scaling, and minimum move thresholds
3. Inventory all requirements R1-R4 and acceptance criteria AC1-AC5, detailing exact numerical thresholds and verification methodologies.
4. Output your comprehensive findings to `D:\backup\Binance data\.agents\teamwork\spec_miner_survey\survey_spec_report.md` and write a standard `handoff.md` in your working directory.
5. When finished, send a message back to parent (4a8e5400-290d-44ec-9df4-e3142d1475c4) with summary and report path.

Constraints: Read-only exploration. Do NOT edit production code or run heavy write operations.
