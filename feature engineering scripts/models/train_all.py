"""
Top-level orchestrator: runs every ensemble member's train.py in sequence,
then the fusion layer (stacking meta-learner -> meta-labeling filter ->
calibration) — one entry point that reproduces the whole 9-member ensemble
+ fusion pipeline end to end, per the plan's Appendix A architecture.

Each stage runs as a subprocess (`python -m models.<member>.train ...` /
`python -m fusion.<stage> ...`) rather than importing and calling run()
directly, so one member's heavy/optional dependencies (torch, neuralforecast,
mamba-ssm, the vendored Kronos repo, a GPU) being unavailable can't crash the
whole run — a failed member is logged and the orchestrator continues with the
rest. Fusion stages depend on each other in order (meta_labeling and
calibration both need stacking_meta_learner's checkpoint), so the fusion
chain stops at the first failure there instead of continuing.

Resume across Kaggle sessions (--resume, on by default): every member and
fusion stage unconditionally writes checkpoints/metrics.json on successful
completion (confirmed by direct inspection of all 9 members + 3 fusion
stages), so before launching a stage this orchestrator checks whether that
file already exists and, if so, skips the stage entirely instead of
retraining it. This is coarse (whole-member) resume, not mid-epoch resume —
the only member with true epoch-level resume is ssm_challenger (its own
checkpoints/inprogress.pt, see models/ssm_challenger/train.py --resume/
--commit-to-hf); the neuralforecast-based members (tft/patchtst/
itransformer/nhits) and kronos_finetune have no upstream mid-epoch resume
mechanism (confirmed via direct source inspection of both libraries), so on
a Kaggle session that gets interrupted partway through one of those, the
member-level skip here is what lets a fresh session pick up where the
LAST-COMPLETED member left off rather than restarting the whole ensemble.
With --upload, a member missing locally is looked up in its private HF repo
(common.hf_model_upload.restore_member): if that repo holds metrics.json the
member's files and its OOF predictions are pulled back down and it is
skipped, so a fresh Kaggle session's empty disk still resumes correctly and
the fusion stages still see every member's OOF. ssm_challenger additionally
gets --commit-to-hf for epoch-level resume. Pass --no-resume (or delete a
member's checkpoints/ dir) to force a clean retrain.

Under a SessionBudget each stage runs with a timeout of the remaining
budget, so no single member runs into Kaggle's hard kill. With --upload,
failures and timeouts are counted on HF (orchestration.checkpoint_manager);
after MAX_MEMBER_ATTEMPTS a member is given up on so the fusion stages —
which only run once every member is finished or given up — and the relay
can complete. A GPU member that fails on 2 GPUs is retried once on 1.

Usage:
    python -m models.train_all --upload
    python -m models.train_all --skip kronos-finetune --val-days 7
    python -m models.train_all --members-only   # skip fusion, e.g. before enough OOF data exists
    python -m models.train_all --gpus 2 --upload   # dual-T4 Kaggle VM, checkpoints committed to HF
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("models.train_all")

MODELS_DIR = Path(__file__).resolve().parent
FUSION_DIR = MODELS_DIR.parent / "fusion"

# (module, member_name, checkpoint_dir, gpu_capable, extra_args). checkpoint_dir
# holds metrics.json on successful completion (confirmed by direct inspection
# of every member's train.py) — that's what --resume checks for. gpu_capable
# marks the members whose train.py accepts --gpus (neuralforecast-based +
# ssm_challenger + kronos_finetune); gbm_family/hmm_regime/sentiment_branch are
# CPU-only (LightGBM/hmmlearn/sklearn) by design, not oversight, so they never
# get --gpus.
MEMBERS: list[tuple[str, str, Path, bool, list[str]]] = [
    ("models.gbm_family.train", "gbm-family", MODELS_DIR / "gbm_family" / "checkpoints", False, []),
    ("models.hmm_regime.train", "hmm-regime", MODELS_DIR / "hmm_regime" / "checkpoints", False, []),
    ("models.sentiment_branch.train", "sentiment-branch", MODELS_DIR / "sentiment_branch" / "checkpoints", False, []),
    ("models.ssm_challenger.train", "ssm-challenger", MODELS_DIR / "ssm_challenger" / "checkpoints", True, []),
    ("models.tft.train", "tft", MODELS_DIR / "tft" / "checkpoints", True, []),
    ("models.patchtst.train", "patchtst", MODELS_DIR / "patchtst" / "checkpoints", True, []),
    ("models.itransformer.train", "itransformer", MODELS_DIR / "itransformer" / "checkpoints", True, []),
    ("models.nhits.train", "nhits", MODELS_DIR / "nhits" / "checkpoints", True, []),
    ("models.kronos_finetune.train", "kronos-finetune", MODELS_DIR / "kronos_finetune" / "checkpoints", True, []),
]

# Order matters: meta_labeling and calibration both load
# stacking_meta_learner's checkpoint, so stacking must run first. Checkpoint
# subdirectory names do NOT uniformly match the module's last segment —
# fusion.meta_labeling writes to checkpoints/meta_labeling_filter/, not
# checkpoints/meta_labeling/ (confirmed via direct source read) — so these
# are given explicitly rather than derived from the module string.
FUSION_STAGES: list[tuple[str, str, Path]] = [
    ("fusion.stacking_meta_learner", "stacking-meta-learner", FUSION_DIR / "checkpoints" / "stacking_meta_learner"),
    ("fusion.meta_labeling", "meta-labeling-filter", FUSION_DIR / "checkpoints" / "meta_labeling_filter"),
    ("fusion.calibration", "calibration", FUSION_DIR / "checkpoints" / "calibration"),
]


# Members that take news_pipeline's text features with --with-news.
NEWS_MEMBERS = {"gbm-family", "sentiment-branch"}


def _hf_name(stage_name: str, is_fusion: bool) -> str:
    # HF repo names of the fusion stages carry a "fusion-" prefix (see each
    # fusion module's MEMBER_NAME / orchestration.checkpoint_manager).
    return f"fusion-{stage_name}" if is_fusion else stage_name


def _cuda_count() -> int:
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:
        return 0


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kills the member and its children (torchrun / Lightning DDP workers)."""
    if os.name == "posix":
        import signal

        for sig, wait in ((signal.SIGTERM, 60), (signal.SIGKILL, 30)):
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                return
            try:
                proc.wait(timeout=wait)
                return
            except subprocess.TimeoutExpired:
                continue
    else:
        proc.kill()
        proc.wait()


def _run(module: str, args: list[str], timeout: float | None = None) -> str:
    """Runs one member/fusion stage; returns "ok", "failed" or "timeout".
    The timeout keeps a single long member from running into Kaggle's hard
    kill — the SessionBudget check only happens between stages."""
    cmd = [sys.executable, "-m", module, *args]
    log.info("Running: %s%s", " ".join(cmd), f" (timeout {timeout / 3600:.2f}h)" if timeout else "")
    proc = subprocess.Popen(cmd, start_new_session=(os.name == "posix"))
    try:
        rc = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        log.error("%s exceeded the session budget — killing it", module)
        _kill_tree(proc)
        return "timeout"
    return "ok" if rc == 0 else "failed"


def _already_done(checkpoint_dir: Path, hf_name: str, restore_from_hf: bool) -> bool:
    if (checkpoint_dir / "metrics.json").exists():
        return True
    if not restore_from_hf:
        return False
    from common.hf_model_upload import restore_member

    if restore_member(hf_name, checkpoint_dir):
        log.info("Restored %s from its HF repo (finished in an earlier session)", hf_name)
        return True
    return False


def _stage_timeout(budget) -> float | None:
    if budget is None:
        return None
    rem = getattr(budget, "remaining_seconds", None)
    if rem is None:
        return None
    margin = getattr(budget, "safety_margin_seconds", 0.0)
    return max(rem() - margin, 60.0)


def run(
    val_days: int,
    max_days: int | None,
    upload: bool,
    skip: list[str],
    members_only: bool,
    gpus: int | None = None,
    resume: bool = True,
    budget=None,
    with_news: bool = False,
    restore_from_hf: bool | None = None,
    track_failures: bool | None = None,
    force_retrain: list[str] | None = None,
) -> dict[str, str]:
    """`restore_from_hf` / `track_failures` default to `upload`: with HF as
    the shared state, a fresh Kaggle session restores finished members
    (weights + OOF) instead of retraining them, and a member that keeps
    failing is given up on after MAX_MEMBER_ATTEMPTS sessions."""
    restore_from_hf = upload if restore_from_hf is None else restore_from_hf
    track_failures = upload if track_failures is None else track_failures
    if track_failures:
        from orchestration.checkpoint_manager import MAX_MEMBER_ATTEMPTS, given_up, load_failures, record_failure

        failures = load_failures()
    else:
        failures = {}

    common_args = ["--val-days", str(val_days)]
    if max_days is not None:
        common_args += ["--max-days", str(max_days)]
    if upload:
        common_args += ["--upload"]
    effective_gpus = gpus if gpus is not None else _cuda_count()

    def member_args(member_name: str, gpu_capable: bool, extra: list[str], gpu_count: int | None) -> list[str]:
        args = common_args + extra
        if gpu_capable and gpu_count is not None:
            args += ["--gpus", str(gpu_count)]
        if member_name == "ssm-challenger" and upload:
            args += ["--commit-to-hf"]  # epoch-level resume across sessions
        if with_news and member_name in NEWS_MEMBERS:
            args += ["--with-news"]
        return args

    def note_failure(name: str, outcome: str) -> None:
        if track_failures:
            count = record_failure(name, outcome)
            log.error("%s: %s (attempt %d/%d)", name, outcome, count, MAX_MEMBER_ATTEMPTS)

    results: dict[str, str] = {}
    stopped_early = False
    force_set = set(force_retrain or [])
    for module, member_name, checkpoint_dir, gpu_capable, extra in MEMBERS:
        if member_name in skip:
            log.info("Skipping %s (--skip)", member_name)
            results[member_name] = "skipped"
            continue
        should_retrain = member_name in force_set or "all" in force_set
        if not should_retrain and resume and _already_done(checkpoint_dir, member_name, restore_from_hf):
            log.info("Skipping %s (--resume: already finished)", member_name)
            results[member_name] = "resumed-skip"
            continue
        if track_failures and given_up(member_name, failures):
            log.warning("Skipping %s — failed %s times already; the ensemble proceeds without it",
                        member_name, failures[member_name]["count"])
            results[member_name] = "given-up"
            continue
        if budget is not None and budget.should_stop():
            log.info("Time budget reached — stopping before %s so this Kaggle session ends cleanly "
                      "(see orchestration/kaggle_relay/). A relaunched session will resume from here.", member_name)
            stopped_early = True
            break
        outcome = _run(module, member_args(member_name, gpu_capable, extra, gpus), _stage_timeout(budget))
        if outcome == "failed" and gpu_capable and effective_gpus > 1 and not (budget and budget.should_stop()):
            # Multi-GPU paths (Lightning DDP predict, torchrun) are the most
            # fragile part of a member; one single-GPU retry before counting it.
            log.warning("%s failed on %d GPUs — retrying once with --gpus 1", member_name, effective_gpus)
            outcome = _run(module, member_args(member_name, gpu_capable, extra, 1), _stage_timeout(budget))
        results[member_name] = outcome
        if outcome != "ok":
            note_failure(member_name, outcome)
            log.error("%s %s — continuing with remaining members", member_name, outcome.upper())

    members_settled = not stopped_early and all(
        results.get(m[1]) in {"ok", "resumed-skip", "skipped", "given-up"} for m in MEMBERS
    )
    if not members_only and not members_settled:
        log.warning("Not every member finished — deferring the fusion stages (they need every member's OOF "
                    "predictions); the next session retries the pending members first")
    elif not members_only:
        fusion_args = ["--val-frac", "0.2"]
        if max_days is not None:
            fusion_args += ["--max-days", str(max_days)]
        if upload:
            fusion_args += ["--upload"]
        for module, stage_name, checkpoint_dir in FUSION_STAGES:
            hf_name = _hf_name(stage_name, is_fusion=True)
            if stage_name in skip:
                log.info("Skipping %s (--skip)", stage_name)
                results[stage_name] = "skipped"
                continue
            should_retrain = stage_name in force_set or hf_name in force_set or "all" in force_set
            if not should_retrain and resume and _already_done(checkpoint_dir, hf_name, restore_from_hf):
                log.info("Skipping %s (--resume: already finished)", stage_name)
                results[stage_name] = "resumed-skip"
                continue
            if track_failures and given_up(hf_name, failures):
                log.warning("Skipping %s and later fusion stages — failed %s times already",
                            stage_name, failures[hf_name]["count"])
                results[stage_name] = "given-up"
                break
            if budget is not None and budget.should_stop():
                log.info("Time budget reached — stopping before %s so this Kaggle session ends cleanly "
                          "(see orchestration/kaggle_relay/). A relaunched session will resume from here.", stage_name)
                break
            outcome = _run(module, fusion_args, _stage_timeout(budget))
            results[stage_name] = outcome
            if outcome != "ok":
                note_failure(hf_name, outcome)
                log.error("%s %s — stopping fusion chain (later stages depend on it)", stage_name, outcome.upper())
                break

    log.info("train_all summary: %s", results)
    failed = [k for k, v in results.items() if v in {"failed", "timeout"}]
    if failed:
        log.error("Failed stages: %s", failed)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-days", type=int, default=config.VAL_DAYS)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--skip", nargs="*", default=[], help="member/fusion-stage names to skip, e.g. kronos-finetune")
    parser.add_argument("--members-only", action="store_true", help="skip the fusion stages")
    parser.add_argument("--gpus", type=int, default=None,
                         help="GPU count threaded into every GPU-capable member's --gpus; default auto-detects per member")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True,
                         help="skip a member/fusion-stage whose checkpoints/metrics.json already exists (default: on)")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                         help="force a clean retrain of every member/fusion-stage regardless of existing checkpoints")
    parser.add_argument("--budget-hours", type=float, default=None,
                         help="stop launching new members/fusion-stages once fewer than 15 minutes remain of this "
                              "many hours (see orchestration/checkpoint_manager.SessionBudget) — for a Kaggle "
                              "session's own proactive stop-before-hard-kill; default None means unrestricted, "
                              "unused outside orchestration/kaggle_relay/run_kaggle_session.py")
    parser.add_argument("--with-news", action="store_true",
                         help="pass --with-news to gbm-family and sentiment-branch (needs a scored news store, "
                              "see news_pipeline.store_sync)")
    parser.add_argument("--restore-from-hf", dest="restore_from_hf", action="store_true", default=None,
                         help="treat members whose HF repo has metrics.json as done and pull their files + OOF "
                              "(default: on with --upload)")
    parser.add_argument("--no-restore-from-hf", dest="restore_from_hf", action="store_false")
    parser.add_argument("--force-retrain", nargs="*", default=[],
                         help="member or fusion stage names to retrain even if checkpoints/HF models exist")
    args = parser.parse_args()
    budget = None
    if args.budget_hours is not None:
        from orchestration.checkpoint_manager import SessionBudget
        budget = SessionBudget(budget_hours=args.budget_hours)
    results = run(args.val_days, args.max_days, args.upload, args.skip, args.members_only, args.gpus, args.resume, budget,
                  with_news=args.with_news, restore_from_hf=args.restore_from_hf, force_retrain=args.force_retrain)
    if any(v in {"failed", "timeout"} for v in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
