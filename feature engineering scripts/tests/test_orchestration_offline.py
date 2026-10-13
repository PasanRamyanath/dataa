"""Offline tests for Gap 5 (orchestration/): SessionBudget, ensemble_complete
(HF calls mocked), models.train_all's budget-stop wiring (subprocess mocked),
and poll_and_relaunch's decision logic (subprocess + ensemble_complete
mocked). No real Kaggle API / HF network calls anywhere in this file."""
import sys
import time
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import config
from orchestration.checkpoint_manager import SessionBudget, ensemble_complete, ALL_MEMBER_NAMES

# ---- Test 1: SessionBudget ----
b = SessionBudget(budget_hours=1.0 / 3600, safety_margin_seconds=0.05)  # 1 second budget, tiny margin
assert not b.should_stop(), "fresh budget should not be expired immediately"
time.sleep(1.0)
assert b.should_stop(), "budget should be expired after sleeping past it"
print("TEST 1 PASSED: SessionBudget expires correctly")

b2 = SessionBudget(budget_hours=10.0, safety_margin_seconds=900.0)
assert not b2.should_stop()
assert b2.remaining_seconds() > 30000
print("TEST 2 PASSED: SessionBudget with real config-like budget stays unexpired")

# ---- Test 3: ensemble_complete with a fully mocked HF API ----
class _FakeApiAllDone:
    def list_repo_files(self, repo_id, repo_type):
        return ["metrics.json", "model.txt"]

complete, status = ensemble_complete(api=_FakeApiAllDone())
assert complete is True
assert all(status.values())
assert set(status.keys()) == set(ALL_MEMBER_NAMES)
print("TEST 3 PASSED: ensemble_complete=True when every repo has metrics.json")


class _FakeApiPartial:
    def list_repo_files(self, repo_id, repo_type):
        if "gbm-family" in repo_id or "hmm-regime" in repo_id:
            return ["metrics.json"]
        raise Exception("repo not found")

complete2, status2 = ensemble_complete(api=_FakeApiPartial())
assert complete2 is False
assert status2["gbm-family"] is True and status2["hmm-regime"] is True
assert status2["tft"] is False
print("TEST 4 PASSED: ensemble_complete=False and per-member status correct when partially trained")

# ---- Test 5: models.train_all budget-stop wiring, with _run mocked so no subprocess ever spawns ----
from models import train_all

calls = []

def _fake_run(module, args, timeout=None):
    calls.append(module)
    return "ok"

class _AlwaysStopBudget:
    def should_stop(self):
        return True

with mock.patch.object(train_all, "_run", _fake_run), \
     mock.patch.object(train_all, "_already_done", return_value=False):
    results = train_all.run(
        val_days=7, max_days=None, upload=False, skip=[], members_only=False,
        gpus=None, resume=True, budget=_AlwaysStopBudget(),
    )

assert calls == [], f"expected zero stages launched with an always-stop budget, got {calls}"
assert all(v not in ("ok", "failed") for v in results.values()), f"no stage should have run: {results}"
print("TEST 5 PASSED: train_all.run() stops before launching any stage when budget.should_stop() is True immediately")


class _NeverStopBudget:
    def should_stop(self):
        return False

calls.clear()
with mock.patch.object(train_all, "_run", _fake_run), \
     mock.patch.object(train_all, "_already_done", return_value=False):
    results2 = train_all.run(
        val_days=7, max_days=None, upload=False, skip=[], members_only=True,  # members_only to keep this test fast/focused
        gpus=None, resume=True, budget=_NeverStopBudget(),
    )

assert len(calls) == len(train_all.MEMBERS), f"expected all {len(train_all.MEMBERS)} members launched with a never-stop budget, got {calls}"
assert all(v == "ok" for v in results2.values())
print("TEST 6 PASSED: train_all.run() launches every member when budget never trips, with fusion skipped via --members-only")

# A budget that trips partway through (stop after 3 calls) -> exactly 3 launched then break
calls.clear()
call_count = {"n": 0}

def _fake_run_counting(module, args, timeout=None):
    calls.append(module)
    return "ok"

class _StopAfterNBudget:
    def __init__(self, n):
        self.n = n
    def should_stop(self):
        return len(calls) >= self.n

with mock.patch.object(train_all, "_run", _fake_run_counting), \
     mock.patch.object(train_all, "_already_done", return_value=False):
    results3 = train_all.run(
        val_days=7, max_days=None, upload=False, skip=[], members_only=True,
        gpus=None, resume=True, budget=_StopAfterNBudget(3),
    )

assert len(calls) == 3, f"expected exactly 3 members launched before budget tripped, got {len(calls)}: {calls}"
n_ok = sum(1 for v in results3.values() if v == "ok")
assert n_ok == 3, f"expected 3 'ok' results, got {results3}"
print("TEST 7 PASSED: train_all.run() stops cleanly mid-loop once budget.should_stop() flips True")

# ---- Test 8: poll_and_relaunch decision logic, subprocess + ensemble_complete mocked ----
import config as cfg
cfg.KAGGLE_USERNAME = "testuser"
cfg.KAGGLE_KERNEL_SLUG = "test-kernel"

from orchestration.kaggle_relay import poll_and_relaunch as par

# Case A: ensemble already complete -> must not even check kernel status or push
with mock.patch("orchestration.kaggle_relay.poll_and_relaunch.ensemble_complete", return_value=(True, {"x": True})), \
     mock.patch("orchestration.kaggle_relay.poll_and_relaunch.kernel_status") as mock_status, \
     mock.patch("orchestration.kaggle_relay.poll_and_relaunch.push_kernel") as mock_push:
    rc = par.main.__wrapped__() if hasattr(par.main, "__wrapped__") else None

# main() parses sys.argv via argparse with no args in this call context, so call it directly
import argparse as _argparse

def run_main(argv):
    old_argv = sys.argv
    sys.argv = ["poll_and_relaunch.py"] + argv
    try:
        return par.main()
    finally:
        sys.argv = old_argv

with mock.patch("orchestration.kaggle_relay.poll_and_relaunch.ensemble_complete", return_value=(True, {"x": True})), \
     mock.patch("orchestration.kaggle_relay.poll_and_relaunch.kernel_status") as mock_status, \
     mock.patch("orchestration.kaggle_relay.poll_and_relaunch.push_kernel") as mock_push:
    rc = run_main([])
    assert rc == 0
    mock_status.assert_not_called()
    mock_push.assert_not_called()
print("TEST 8 PASSED: poll_and_relaunch no-ops (no status check, no push) when ensemble already complete")

# Case B: ensemble incomplete, kernel status 'running' -> no push
with mock.patch("orchestration.kaggle_relay.poll_and_relaunch.ensemble_complete", return_value=(False, {"tft": False})), \
     mock.patch("orchestration.kaggle_relay.poll_and_relaunch.kernel_status", return_value="running"), \
     mock.patch("orchestration.kaggle_relay.poll_and_relaunch.push_kernel") as mock_push:
    rc = run_main([])
    assert rc == 0
    mock_push.assert_not_called()
print("TEST 9 PASSED: poll_and_relaunch does not push while kernel is still running")

# Case C: ensemble incomplete, kernel status 'complete' -> pushes
with mock.patch("orchestration.kaggle_relay.poll_and_relaunch.ensemble_complete", return_value=(False, {"tft": False})), \
     mock.patch("orchestration.kaggle_relay.poll_and_relaunch.kernel_status", return_value="complete"), \
     mock.patch("orchestration.kaggle_relay.poll_and_relaunch.push_kernel", return_value=True) as mock_push:
    rc = run_main([])
    assert rc == 0
    mock_push.assert_called_once()
print("TEST 10 PASSED: poll_and_relaunch relaunches when the last session ended but the ensemble isn't done")

# Case D: ensemble incomplete, kernel never pushed (status None), no --allow-first-push -> no push
with mock.patch("orchestration.kaggle_relay.poll_and_relaunch.ensemble_complete", return_value=(False, {"tft": False})), \
     mock.patch("orchestration.kaggle_relay.poll_and_relaunch.kernel_status", return_value=None), \
     mock.patch("orchestration.kaggle_relay.poll_and_relaunch.push_kernel") as mock_push:
    rc = run_main([])
    assert rc == 0
    mock_push.assert_not_called()
print("TEST 11 PASSED: poll_and_relaunch does not auto-launch a never-pushed kernel without --allow-first-push")

# Case E: same as D but with --allow-first-push -> pushes
with mock.patch("orchestration.kaggle_relay.poll_and_relaunch.ensemble_complete", return_value=(False, {"tft": False})), \
     mock.patch("orchestration.kaggle_relay.poll_and_relaunch.kernel_status", return_value=None), \
     mock.patch("orchestration.kaggle_relay.poll_and_relaunch.push_kernel", return_value=True) as mock_push:
    rc = run_main(["--allow-first-push"])
    assert rc == 0
    mock_push.assert_called_once()
print("TEST 12 PASSED: poll_and_relaunch performs first launch with --allow-first-push")

# Case F: --dry-run never calls push_kernel even when it would otherwise push
with mock.patch("orchestration.kaggle_relay.poll_and_relaunch.ensemble_complete", return_value=(False, {"tft": False})), \
     mock.patch("orchestration.kaggle_relay.poll_and_relaunch.kernel_status", return_value="error"), \
     mock.patch("orchestration.kaggle_relay.poll_and_relaunch.push_kernel") as mock_push:
    rc = run_main(["--dry-run"])
    assert rc == 0
    mock_push.assert_not_called()
print("TEST 13 PASSED: --dry-run never calls push_kernel")

# ---- Test 14: kernel_status parsing + push_kernel, with subprocess.run itself mocked (no real `kaggle` CLI call) ----
class _FakeCompleted:
    def __init__(self, returncode, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr

with mock.patch("orchestration.kaggle_relay.poll_and_relaunch.subprocess.run",
                return_value=_FakeCompleted(0, stdout='testuser/test-kernel has status "complete"')) as mock_run:
    st = par.kernel_status("testuser/test-kernel")
    assert st == "complete", f"expected parsed status 'complete', got {st!r}"
    assert mock_run.call_args.args[0][:3] == ["kaggle", "kernels", "status"]
print("TEST 14 PASSED: kernel_status correctly parses the kaggle CLI's quoted status output")

with mock.patch("orchestration.kaggle_relay.poll_and_relaunch.subprocess.run",
                return_value=_FakeCompleted(1, stderr="404 not found")) as mock_run:
    st = par.kernel_status("testuser/never-pushed")
    assert st is None
print("TEST 15 PASSED: kernel_status returns None on a non-zero exit (kernel not found / never pushed)")

import tempfile, json
tmp_dir = Path(tempfile.mkdtemp(prefix="kaggle_relay_test_"))
with mock.patch("orchestration.kaggle_relay.poll_and_relaunch.subprocess.run",
                return_value=_FakeCompleted(0, stdout="Kernel version 3 successfully pushed")) as mock_run:
    ok = par.push_kernel(tmp_dir)
    assert ok is True
    meta = json.loads((tmp_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta["id"] == "testuser/test-kernel"
    assert meta["code_file"] == par.BUNDLE_NAME
    assert mock_run.call_args.args[0] == ["kaggle", "kernels", "push", "-p", str(tmp_dir)]
print("TEST 16 PASSED: push_kernel renders kernel-metadata.json from config and calls the kaggle CLI correctly")
import shutil
shutil.rmtree(tmp_dir, ignore_errors=True)

print("ALL ORCHESTRATION OFFLINE TESTS PASSED")
