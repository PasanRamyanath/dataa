"""
External half of Kaggle session-relaunch orchestration (§11.2). Runs OFF
Kaggle — on a GitHub Actions cron (see .github/workflows/kaggle_relay.yml) —
polling the Kaggle API for the ensemble-training kernel's status and
re-pushing it whenever a session has ended (finished, crashed, cancelled, or
hit Kaggle's ~12h ceiling) but the ensemble isn't fully trained yet, per
orchestration.checkpoint_manager.ensemble_complete() (which asks each
member/fusion-stage's private HF model repo, not any Kaggle-local state,
since a fresh Kaggle session's disk starts empty). This closes the loop with
orchestration/checkpoint_manager.SessionBudget, the Kaggle-side half that
makes each session stop cleanly (committing its in-progress stage's
checkpoint to HF first) a safety margin before the hard kill, so this poller
should always find a *cleanly finished* session to relaunch rather than
racing one that got killed mid-write.

Talks to Kaggle exclusively through the official `kaggle` CLI via
subprocess — never imports the `kaggle` python package directly — so
credentials stay confined to the KAGGLE_USERNAME/KAGGLE_KEY env vars the CLI
itself reads (exactly as documented for a GitHub Actions runner), and every
test of this module monkeypatches subprocess.run rather than touching a real
Kaggle account. This script is NEVER run against a real Kaggle account from
within this development environment — only from the scheduled GitHub
Actions workflow, with real Kaggle secrets present.

The exact lowercase status strings the `kaggle kernels status` CLI prints
("complete", "error", "running", "queued", "cancelAcknowledged", ...) were
not independently re-verified against a live call while building this
(no-live-network-calls policy) — they're the CLI's documented/observed
convention. Re-check `kaggle kernels status <ref>` output once against a
real kernel before this workflow's first live use, and adjust
ACTIVE_STATUSES/ENDED_STATUSES if it differs.

Usage (as run by the GitHub Actions workflow):
    python -m orchestration.kaggle_relay.poll_and_relaunch
    python -m orchestration.kaggle_relay.poll_and_relaunch --dry-run
    python -m orchestration.kaggle_relay.poll_and_relaunch --allow-first-push
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import config
from orchestration.kaggle_relay import build_bundle
from orchestration.kaggle_relay.build_bundle import BUNDLE_NAME
from orchestration.checkpoint_manager import ensemble_complete

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("orchestration.kaggle_relay.poll_and_relaunch")

# "newscript" (KernelWorkerStatus.NEW_SCRIPT): a just-pushed version not yet
# queued — relaunching it would only stack a second version on top.
ACTIVE_STATUSES = {"running", "queued", "newscript"}
ENDED_STATUSES = {"complete", "error", "cancelacknowledged", "cancelrequested"}

def kernel_ref() -> str:
    if not config.KAGGLE_USERNAME:
        raise RuntimeError("KAGGLE_USERNAME is not set — required to address the training kernel")
    return f"{config.KAGGLE_USERNAME}/{config.KAGGLE_KERNEL_SLUG}"


def _ensure_kaggle_env() -> dict[str, str]:
    env = dict(os.environ)
    search_paths = [
        Path(r"D:\backup\Binance data\feature engineering scripts\kaggle.json"),
        Path(r"d:\backup\feature engineering scripts\kaggle.json"),
        Path(__file__).resolve().parent.parent.parent / "kaggle.json",
        config.PROJECT_ROOT / "feature engineering scripts" / "kaggle.json",
        config.PROJECT_ROOT / "kaggle.json",
        Path.home() / ".kaggle" / "kaggle.json",
    ]
    for p in search_paths:
        if p.exists():
            try:
                import json
                d = json.loads(p.read_text(encoding="utf-8"))
                if d.get("username") and d.get("key"):
                    env["KAGGLE_USERNAME"] = d["username"]
                    env["KAGGLE_KEY"] = d["key"]
                    break
            except Exception:
                pass
    return env


def kernel_status(ref: str) -> str | None:
    """Returns the kernel's lowercased status string, or None if the kernel
    has never been pushed (first run) or the CLI call otherwise fails —
    treated as 'nothing to relaunch yet', not an error, since a first push
    is a deliberate, separate action (see --allow-first-push)."""
    try:
        result = subprocess.run(
            ["kaggle", "kernels", "status", ref],
            capture_output=True, text=True, timeout=60, env=_ensure_kaggle_env(),
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        log.warning("kaggle CLI call failed: %s", e)
        return None
    if result.returncode != 0:
        log.info("kernel status check failed (exit %d) — likely not pushed yet: %s",
                  result.returncode, (result.stderr or "").strip())
        return None
    out = (result.stdout or "").strip()
    log.info("kaggle kernels status output: %s", out)
    if '"' in out:
        out = out.split('"')[1]
    return normalize_status(out)


def normalize_status(raw: str) -> str | None:
    """Newer CLI versions print the enum form ("KernelWorkerStatus.RUNNING",
    "CANCEL_ACKNOWLEDGED") where older ones printed "running" /
    "cancelAcknowledged" — reduce both to the bare lowercase name, so a
    running kernel is never mistaken for an unrecognized (=> relaunched) one."""
    name = raw.strip().rsplit(".", 1)[-1].replace("_", "").lower()
    return name or None


def write_kernel_metadata(target_dir: Path) -> None:
    """Renders kernel-metadata.json from config right before a push, so the
    user's actual Kaggle username never sits hardcoded in a committed file."""
    metadata = {
        "id": kernel_ref(),
        "title": config.KAGGLE_KERNEL_SLUG,
        "code_file": BUNDLE_NAME,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    if config.KAGGLE_MACHINE_SHAPE:
        metadata["machine_shape"] = config.KAGGLE_MACHINE_SHAPE
    (target_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def push_kernel(kernel_dir: Path | None = None) -> bool:
    """Kaggle receives only the kernel's code_file, so every push ships a
    freshly built single-file bundle of the current code (build_bundle.py)
    from a staging dir, never this package dir itself."""
    if kernel_dir is None:
        kernel_dir = Path(tempfile.mkdtemp(prefix="kaggle_kernel_"))
    build_bundle.build(kernel_dir / BUNDLE_NAME)
    write_kernel_metadata(kernel_dir)
    try:
        result = subprocess.run(
            ["kaggle", "kernels", "push", "-p", str(kernel_dir)],
            capture_output=True, text=True, timeout=300, env=_ensure_kaggle_env(),
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        log.error("kaggle kernels push failed to run: %s", e)
        return False
    if result.returncode != 0:
        log.error("kaggle kernels push failed (exit %d): %s | %s",
                  result.returncode, (result.stdout or "").strip(), (result.stderr or "").strip())
        return False
    log.info("kaggle kernels push succeeded: %s", (result.stdout or "").strip())
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="log the decision but never call `kaggle kernels push`")
    parser.add_argument("--allow-first-push", action="store_true",
                         help="push even if the kernel has never been pushed before (default: the very first "
                              "launch is left to a human, since it starts consuming Kaggle GPU quota)")
    parser.add_argument("--kernel-dir", default=None,
                         help="staging dir for the bundle + kernel-metadata.json (default: a fresh temp dir)")
    args = parser.parse_args()

    complete, status = ensemble_complete()
    if complete:
        log.info("Ensemble fully trained (all %d members/fusion stages have metrics.json) — nothing to relaunch", len(status))
        return 0
    pending = [name for name, done in status.items() if not done]
    log.info("%d/%d member(s)/fusion-stage(s) still pending: %s", len(pending), len(status), pending)

    ref = kernel_ref()
    st = kernel_status(ref)

    should_push = False
    if st is None:
        if args.allow_first_push:
            log.info("No existing kernel found — performing first launch (--allow-first-push)")
            should_push = True
        else:
            log.info("Kernel %s has no status yet (never pushed) — skipping; pass --allow-first-push "
                      "to let this poller perform the very first launch", ref)
    elif st in ACTIVE_STATUSES:
        log.info("Kernel %s is still %s — nothing to do", ref, st)
    else:
        if st not in ENDED_STATUSES:
            log.warning("Kernel %s has unrecognized status %r — treating as ended and relaunching", ref, st)
        else:
            log.info("Kernel %s has ended (status=%s) and the ensemble isn't done — relaunching", ref, st)
        should_push = True

    if not should_push:
        return 0
    if args.dry_run:
        log.info("--dry-run: would push kernel %s now", ref)
        return 0

    ok = push_kernel(Path(args.kernel_dir) if args.kernel_dir else None)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
