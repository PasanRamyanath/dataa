## 2026-10-18T01:21:15Z
[Message] timestamp=2026-10-18T01:21:15Z sender=4a8e5400-290d-44ec-9df4-e3142d1475c4 priority=MESSAGE_PRIORITY_HIGH content=You are the Features & Models Explorer for the BTCUSDT 10-Second Alpha & Execution Overhaul project.

Your Identity & Working Directory:
- Working Directory: D:\backup\Binance data\.agents\teamwork\explorer_models_survey
- Parent Conversation ID: 4a8e5400-290d-44ec-9df4-e3142d1475c4

MANDATORY FIRST STEP:
Read D:\backup\Binance data\.agents\teamwork\ORIGINAL_REQUEST.md completely.

Your Mission:
Conduct an exhaustive codebase investigation into the current feature engineering, data pipelines, model ensemble, and meta-labeler:
1. Locate and inspect all current feature calculation code, data loaders, and caching (e.g. 174-feature set, parquet/csv files, trade flow, order book, sentiment).
2. Pinpoint the exact source of intra-bar lookahead leakage in the sentiment/micro-signal branch (`_live_micro_signal` reading unclosed bar returns) and any other lookahead leaks across all features.
3. Inspect the member models (NHITS, PatchTST, TFT, SSM, or any other predictors) and analyze why their predictions are collinear (r > 0.98).
4. Inspect the meta-labeler implementation: locate where artificial threshold clamping occurs (e.g. `p_sel < 0.62` clamped to 0.42) and how confidence probabilities are currently computed.
5. Inventory available data sources in the workspace (ETHUSDT data, spot vs perp data, order book L1/L2 data, liquidations, funding rates) to support the required orthogonal feature families (Cross-asset lead-lag, Spot-perp basis/funding, Order book microstructure/queue imbalance/VPIN).
6. Output your detailed findings to `D:\backup\Binance data\.agents\teamwork\explorer_models_survey\survey_models_report.md` and write a standard `handoff.md` in your working directory.
7. When finished, send a message back to parent (4a8e5400-290d-44ec-9df4-e3142d1475c4) with summary and report path.

Constraints: Read-only exploration. Do NOT edit production code or train models.
