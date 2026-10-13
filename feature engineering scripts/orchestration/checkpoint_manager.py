"""
Kaggle-side half of session-relaunch orchestration (§11.2) — the piece
models/train_all.py's existing --resume (per-member/per-fusion-stage
metrics.json skip) doesn't cover on its own: knowing *when* to stop
launching new stages within a single session, and knowing *whether the whole
ensemble is already done* so the external relaunch loop (see
orchestration/kaggle_relay/poll_and_relaunch.py) can eventually stop too.

SessionBudget is a plain wall-clock guard threaded into models.train_all.run()
via its optional `budget` parameter (see orchestration/kaggle_relay/
run_kaggle_session.py, the Kaggle-side entry point that actually uses it):
before launching each member/fusion-stage subprocess, train_all checks
budget.should_stop() and, if the remaining time is under the safety margin,
stops cleanly instead of risking Kaggle's hard ~12h kill landing mid-write.
This mirrors kaggle_run_pipeline.py's existing per-day time-budget pattern,
just applied at member/fusion-stage granularity instead of per-day.

ensemble_complete() answers "is there anything left to train at all" by
asking each member/fusion-stage's own private HF model repo for a
metrics.json (the same completion signal train_all.py's local --resume
check uses, just read from HF instead of local disk) — this is what a
GitHub Actions poller with no access to any Kaggle-local filesystem can use
to decide it should stop relaunching the kernel.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("orchestration.checkpoint_manager")

# Cross-session failure ledger: a member that fails (or overruns a session)
# this many times is given up on, so the fusion stages can proceed without
# it and the relay stops relaunching instead of burning GPU quota forever.
MAX_MEMBER_ATTEMPTS = int(os.environ.get("MAX_MEMBER_ATTEMPTS", "3"))
FAILURES_PATH_IN_REPO = "run_state/failures.json"

# Mirrors models/train_all.py's MEMBERS + FUSION_STAGES, but using the HF
# model-repo MEMBER_NAME strings (confirmed via direct source read of every
# member's train.py and every fusion stage) rather than train_all's local
# stage_name strings, which differ for the fusion stages (no "fusion-"
# prefix locally, e.g. "stacking-meta-learner" vs. the HF repo's
# "fusion-stacking-meta-learner").
ALL_MEMBER_NAMES: list[str] = [
    "gbm-family", "hmm-regime", "sentiment-branch", "ssm-challenger",
    "tft", "patchtst", "itransformer", "nhits", "kronos-finetune",
    "fusion-stacking-meta-learner", "fusion-meta-labeling-filter", "fusion-calibration",
]


class SessionBudget:
    """Wall-clock guard for a single Kaggle session. `should_stop()` goes
    True once fewer than `safety_margin_seconds` remain in the budget, giving
    the caller (models.train_all.run()) a chance to stop launching new
    stages *before* the current one would risk still being mid-write when
    Kaggle kills the VM — training-run resume (checkpoints already committed
    by the last-completed stage) then covers the rest in the next session."""

    def __init__(self, budget_hours: float | None = None, safety_margin_seconds: float = 900.0):
        self.budget_seconds = (budget_hours if budget_hours is not None else config.KAGGLE_SESSION_TIME_BUDGET_HOURS) * 3600.0
        self.safety_margin_seconds = safety_margin_seconds
        self._start = time.monotonic()

    def elapsed_seconds(self) -> float:
        return time.monotonic() - self._start

    def remaining_seconds(self) -> float:
        return self.budget_seconds - self.elapsed_seconds()

    def should_stop(self) -> bool:
        return self.remaining_seconds() <= self.safety_margin_seconds


def _member_has_metrics(api, member_name: str) -> bool:
    from common.hf_model_upload import model_repo_id

    try:
        files = api.list_repo_files(repo_id=model_repo_id(member_name), repo_type="model")
    except Exception:
        return False
    return "metrics.json" in files


def load_failures(api=None) -> dict[str, dict]:
    """{member_name: {"count": n, "last_reason": str}} from the private
    training-data repo; {} when absent or unreachable."""
    from huggingface_hub import hf_hub_download

    from common.hf_model_upload import get_api

    try:
        api = api or get_api()
        path = hf_hub_download(repo_id=config.HF_TRAINING_DATA_REPO_ID, filename=FAILURES_PATH_IN_REPO,
                               repo_type="dataset", token=config.get_hf_token())
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def record_failure(member_name: str, reason: str, api=None) -> int:
    """Increments the member's failure count on HF and returns the new count
    (best effort: logs and returns the local count if HF is unreachable)."""
    from common.hf_model_upload import ensure_private_repo, get_api

    failures = load_failures(api)
    entry = failures.get(member_name, {"count": 0})
    entry = {"count": int(entry.get("count", 0)) + 1, "last_reason": reason}
    failures[member_name] = entry
    try:
        api = api or get_api()
        ensure_private_repo(config.HF_TRAINING_DATA_REPO_ID, "dataset", api)
        api.upload_file(path_or_fileobj=json.dumps(failures, indent=2).encode("utf-8"),
                        path_in_repo=FAILURES_PATH_IN_REPO, repo_id=config.HF_TRAINING_DATA_REPO_ID,
                        repo_type="dataset", commit_message=f"{member_name}: failure {entry['count']} ({reason})")
    except Exception as e:
        log.warning("Could not record failure of %s on HF: %s", member_name, e)
    return entry["count"]


def given_up(member_name: str, failures: dict[str, dict]) -> bool:
    return int(failures.get(member_name, {}).get("count", 0)) >= MAX_MEMBER_ATTEMPTS


def ensemble_complete(api=None) -> tuple[bool, dict[str, bool]]:
    """Returns (all_done, {member_name: done}). A member counts as done when
    its repo has metrics.json or it has used up MAX_MEMBER_ATTEMPTS. Never
    raises — a repo that doesn't exist yet or any other HF API failure for
    one member is treated as 'not done yet' for that member."""
    from common.hf_model_upload import get_api

    api = api or get_api()
    failures = load_failures(api)
    status = {name: _member_has_metrics(api, name) or given_up(name, failures) for name in ALL_MEMBER_NAMES}
    return all(status.values()), status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-complete", action="store_true",
                         help="print per-member/fusion-stage HF checkpoint status and exit 0 if the whole "
                              "ensemble is done, 1 otherwise")
    args = parser.parse_args()
    if not args.check_complete:
        parser.print_help()
        return
    complete, status = ensemble_complete()
    for name, done in status.items():
        log.info("%-32s %s", name, "done" if done else "pending")
    log.info("Ensemble complete: %s", complete)
    sys.exit(0 if complete else 1)


if __name__ == "__main__":
    main()
