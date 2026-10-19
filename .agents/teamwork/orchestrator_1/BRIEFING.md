# BRIEFING — 2026-10-18T01:20:00Z

## Mission
Orchestrate the complete architectural and algorithmic overhaul of the BTCUSDT 10-second-bar futures prediction and maker execution pipeline, satisfying R1-R4 and AC1-AC5.

## 🔒 My Identity
- Archetype: orchestrator
- Roles: orchestrator, user_liaison, human_reporter, successor
- Working directory: D:\backup\Binance data\.agents\teamwork\orchestrator_1
- Original parent: Sentinel
- Original parent conversation ID: 8c514ff2-17ee-4d1c-8bdd-24e7dcaaee65

## 🔒 My Workflow
- **Pattern**: Project Pattern (Dual Track: Implementation Track + E2E Testing Track)
- **Scope document**: D:\backup\Binance data\PROJECT.md
1. **Decompose**: Survey full scope with 3 Explorers/Spec Miners, synthesize feature inventory, decompose into milestones (M1: Feature de-duplication, orthogonal signals & leakage elimination; M2: Continuous calibrated meta-labeler & volatility hurdle; M3: BBO queue-aware maker execution node & barrier sync; M4: Purged walk-forward evaluation, live telemetry harness & reproducible test suite; M5: Final E2E test pass & adversarial coverage hardening).
2. **Dispatch & Execute**: Direct iteration loop or delegate to sub-orchestrators for complex milestones.
   - For direct milestones: 3 Explorers -> 1 Worker -> 2 Reviewers -> 2 Challengers -> 1 Auditor -> Gate check.
   - For E2E testing track: spawn dedicated E2E Testing Orchestrator / workers in parallel.
3. **On failure**:
   - Retry: nudge stuck agent or re-send task
   - Replace: spawn fresh agent with partial progress
   - Skip: proceed without (only if non-critical, NEVER skip auditor)
   - Redistribute: split stuck agent's remaining work
   - Redesign: re-partition decomposition
   - Escalate: report to parent (Project Orchestrator redesigns)
4. **Succession**: At 16 spawns, write handoff.md, spawn successor.
- **Work items**:
  1. Survey & Map Scope [in-progress]
  2. Architecture Decomposition & PROJECT.md [pending]
  3. Milestone 1: Orthogonal Multi-Architecture Ensemble & Feature De-Duplication (R1, AC1, AC3) [pending]
  4. Milestone 2: Continuous Calibrated Meta-Labeler & Volatility Hurdle Engine (R2, AC2) [pending]
  5. Milestone 3: BBO Queue-Aware Maker Execution Node & Dynamic Barriers (R3, AC4) [pending]
  6. Milestone 4: Purged Walk-Forward Evaluation & Live Telemetry Harness (R4, AC5) [pending]
  7. Milestone 5: E2E Verification & Adversarial Coverage Hardening (AC1-AC5) [pending]
- **Current phase**: 0 (Survey & Scope Mapping)
- **Current focus**: Surveying codebase, empirical ledger, existing scripts, and data sources

## 🔒 Key Constraints
- NEVER write, modify, or create source code files directly.
- NEVER run build/test commands yourself — require workers to do so.
- NEVER investigate or explore the problem at the code level — dispatch Explorers for technical investigation.
- Use file-editing tools ONLY for metadata/state files (.md) in .agents/teamwork/.
- Never reuse a subagent after it has delivered its handoff — always spawn fresh.
- Binary veto on Forensic Auditor integrity violations.

## Current Parent
- Conversation ID: 8c514ff2-17ee-4d1c-8bdd-24e7dcaaee65
- Updated: not yet

## Key Decisions Made
- Initialized Project Orchestrator state. Initiating parallel 3-agent survey phase per Project Pattern.

## Team Roster
| Agent | Type | Work Item | Status | Conv ID |
|---|---|---|---|---|
| spec_miner_survey | teamwork_preview_spec_miner | Survey 1: Spec & Ledger Mining | completed | 8c1bfd30-13dc-4d09-b51f-c484c1e0361b |
| explorer_models_survey | teamwork_preview_explorer | Survey 2: Features & Models Exploration | completed | 68449b74-9185-468e-a9cd-4c62f4428b6d |
| explorer_exec_survey | teamwork_preview_explorer | Survey 3: Execution & Backtest Exploration | completed | 80b677eb-a3a5-420e-a33d-f61049320276 |
| test_writer_e2e | teamwork_preview_test_writer | E2E Testing Track: TEST_INFRA & Tiers 1-4 | completed | 241b32c8-821d-42db-9d43-700cb3a026fd |
| explorer_m1_leakage | teamwork_preview_explorer | M1 Explorer: Anti-Leakage Pipeline Strategy | completed | ec830bf1-d676-4f4c-afd9-f5477ceb3e34 |
| explorer_m1_features | teamwork_preview_explorer | M1 Explorer: Orthogonal Features Strategy | completed | 9efc0e98-11f5-4fda-bdb0-aa8a4564ff18 |
| explorer_m1_ensemble | teamwork_preview_explorer | M1 Explorer: Ensemble De-Correlation Strategy | completed | afbb924e-ea51-4dff-ae94-8cccfee69f96 |
| worker_m1 | teamwork_preview_worker | Milestone 1 Implementation (R1, AC1, AC3) | completed | 4b113623-6ab6-4a10-8f5e-ea9c2bfc4d7a |
| reviewer_m1_1 | teamwork_preview_reviewer | Milestone 1 Review: Contracts & Correctness | in-progress | 0676593e-5d91-40d0-8fd2-36544d229b67 |
| reviewer_m1_2 | teamwork_preview_reviewer | Milestone 1 Review: Robustness & Regressions | in-progress | beef1d54-6f96-47cf-823f-3c9b84dd4de0 |
| challenger_m1_1 | teamwork_preview_challenger | Milestone 1 Challenge: Anti-Leakage Stress | in-progress | 45127628-a8ce-4754-967c-a6d2c52e16e2 |
| challenger_m1_2 | teamwork_preview_challenger | Milestone 1 Challenge: De-Correlation Stress | completed | e801801b-5474-459a-93c0-f3d698e23f62 |
| auditor_m1_1 | teamwork_preview_auditor | Milestone 1 Forensic Integrity Audit | in-progress | 99c60b6e-e4e5-4bba-a010-68c58a3ee404 |

## Succession Status
- Succession required: no
- Spawn count: 13 / 16
- Pending subagents: 0676593e-5d91-40d0-8fd2-36544d229b67, beef1d54-6f96-47cf-823f-3c9b84dd4de0, 45127628-a8ce-4754-967c-a6d2c52e16e2, 99c60b6e-e4e5-4bba-a010-68c58a3ee404
- Predecessor: none
- Successor: not yet spawned

## Active Timers
- Heartbeat cron: 4a8e5400-290d-44ec-9df4-e3142d1475c4/task-10
- Safety timer: none
- On succession: kill all timers before spawning successor
- On context truncation: run `manage_task(Action="list")` — re-create if missing

## Artifact Index
- D:\backup\Binance data\.agents\teamwork\ORIGINAL_REQUEST.md — Authoritative user requirements
- D:\backup\Binance data\.agents\teamwork\orchestrator_1\DISPATCH.md — Dispatch log
- D:\backup\Binance data\.agents\teamwork\orchestrator_1\BRIEFING.md — Persistent working memory
- D:\backup\Binance data\.agents\teamwork\orchestrator_1\progress.md — Liveness & progress tracker
