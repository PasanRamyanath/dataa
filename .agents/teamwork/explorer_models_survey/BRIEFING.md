# BRIEFING — 2026-10-18T01:30:00Z

## Mission
Exhaustive codebase investigation into current feature engineering, data pipelines, model ensemble, meta-labeler, lookahead leakage, model collinearity, and available data sources for BTCUSDT 10-second alpha.

## 🔒 My Identity
- Archetype: explorer
- Roles: Features & Models Explorer
- Working directory: D:\backup\Binance data\.agents\teamwork\explorer_models_survey
- Original parent: 4a8e5400-290d-44ec-9df4-e3142d1475c4
- Milestone: Milestone 1 - Architectural Survey & Gap Analysis

## 🔒 Key Constraints
- Read-only investigation — do NOT implement
- Do NOT edit production code or train models
- Focus on feature engineering, data pipelines, model ensemble, meta-labeler, lookahead leakage, collinearity, data inventory

## Current Parent
- Conversation ID: 4a8e5400-290d-44ec-9df4-e3142d1475c4
- Updated: 2026-10-18T01:30:00Z

## Investigation State
- **Explored paths**:
  - `feature engineering scripts/inference/checkpoint_loaders.py`
  - `feature engineering scripts/streaming/live_loop_v2.py`
  - `feature engineering scripts/streaming/live_ingestion/binance_ws.py`
  - `feature engineering scripts/streaming/live_ingestion/ring_buffer.py`
  - `feature engineering scripts/common/feature_contract.py`
  - `feature engineering scripts/fusion/meta_labeling.py`
  - `feature engineering scripts/fusion/stacking_meta_learner.py`
  - `feature engineering scripts/fusion/oof/*.parquet`
  - `feature engineering scripts/scratch/rigorous_evaluation_audit.py`
  - `scratch/live_45m_eval_ledger.jsonl`
- **Key findings**:
  - Intra-bar forward lookahead leakage in `binance_ws.py:get_live_snapshot` and `live_loop_v2.py:_fuse`: unclosed $t+1$ bar returns during GPU latency are injected into `_live_micro_signal` and Bayesian probability adjustment.
  - Root cause of member collinearity ($r > 0.98$): all predictors in `checkpoint_loaders.py` blend 30% to 50% `_live_micro_signal`, and 4 transformer models share the identical `_NeuralForecastPredictor` wrapper and `_signal_to_3class_proba` mapping. In raw OOF parquets, correlation is only 0.05 to 0.52.
  - Meta-labeler bypass & threshold clamping: trained `MetaLabelFilter` is loaded but never executed in `live_loop_v2.py`; replaced with hardcoded rules clamping flat bars to 0.42 and non-conforming bars to [0.30, 0.52], causing bimodal collapse and ECE=20.74%.
  - Data sources: BTCUSDT perp/spot, bookTicker, bookDepth, and aggTrades are fully ready; ETHUSDT cross-asset signals can be streamed via WebSocket or fetched by expanding downloader scripts.
- **Unexplored areas**: None for survey scope. Survey is complete.

## Key Decisions Made
- Fully documented all 5 investigation areas in `survey_models_report.md`
- Completed 5-component `handoff.md`
- Prepared execution summary for orchestrator

## Artifact Index
- DISPATCH.md — Incoming task dispatch record
- BRIEFING.md — Persistent working memory
- progress.md — Liveness heartbeat and progress tracker
- survey_models_report.md — Comprehensive survey report
- handoff.md — 5-component handoff report
