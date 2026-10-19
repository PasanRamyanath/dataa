# QA Handoff Skill (Local Copy)
Source: C:\Users\pasan\.gemini\config\skills\qa-handoff\SKILL.md

Generates a QA handoff document (markdown) for implemented features/fixes, ready to hand to the QA team. Investigates git history, diffs, test files/results, TODOs, and known issues to fill out a fixed 8-section template: summary, environment, test scenarios, impacted areas, test data, known issues, out of scope, and screenshots/logs.

Hard rules:
- Never write secrets (passwords, tokens, API keys, connection strings) into Section 5 or anywhere else.
- Only run test scripts that are read-only / non-destructive against dev-safe targets.
- If a section has no evidence, write "Not applicable" or "No known issues at this time".
- QA-friendly language: plain, step-by-step, no internal jargon.
- Ask for target scope if ambiguous.
