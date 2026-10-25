"""
feature engineering scripts/orchestration/checkpoint_manager_v4.py
Fault-Tolerant Dual-T4 GPU & CPU Wave Orchestrator with Per-Model Hugging Face
Checkpoint Resume & 8.5-Hour Soft-Stop Budget Guard (new_plan.md §22.1 - §22.4).

Manages all 15 multi-horizon base ensemble members across Tier 4 (Macro),
Tier 3 (Meso), Tier 2 (Tactical), and Tier 1 (Micro Sniper), plus Stage 5
Supervised Hierarchical Fusion (`fusion-v4-hierarchical`).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any, Callable

from common.hf_model_upload import (
    canonical_v4_member_name,
    download_all_completed_members,
    is_member_completed_on_hf,
    resolve_hf_token,
    upload_fusion_artifacts,
    upload_member_checkpoint,
    v4_tier_repo_id,
)

log = logging.getLogger("orchestration.checkpoint_manager_v4")


@dataclass(frozen=True)
class MultiHorizonMemberSpec:
    member_name: str
    tier: str                      # "tier4_macro", "tier3_meso", "tier2_tactical", "tier1_micro"
    preferred_device: str          # "cuda:0", "cuda:1", or "cpu"
    estimated_minutes_t4: float
    requires_oof_parquet: bool = True


MULTI_HORIZON_MEMBER_REGISTRY: tuple[MultiHorizonMemberSpec, ...] = (
    # Tier 4 Macro (4 models)
    MultiHorizonMemberSpec("tier4-hmm-regime",       "tier4_macro",    "cpu",    4.0),
    MultiHorizonMemberSpec("tier4-patchtst-ffd",     "tier4_macro",    "cuda:0", 14.0),
    MultiHorizonMemberSpec("tier4-tft-funding",      "tier4_macro",    "cuda:0", 16.0),
    MultiHorizonMemberSpec("tier4-kronos-macro",     "tier4_macro",    "cuda:1", 22.0),
    # Tier 3 Meso (4 models)
    MultiHorizonMemberSpec("tier3-lgbm-positioning", "tier3_meso",     "cpu",    8.0),
    MultiHorizonMemberSpec("tier3-itransformer-32",  "tier3_meso",     "cuda:0", 20.0),
    MultiHorizonMemberSpec("tier3-nhits-multirate",  "tier3_meso",     "cuda:1", 18.0),
    MultiHorizonMemberSpec("tier3-kronos-meso",      "tier3_meso",     "cuda:1", 22.0),
    # Tier 2 Tactical (4 models)
    MultiHorizonMemberSpec("tier2-lgbm-bookdepth",   "tier2_tactical", "cpu",    10.0),
    MultiHorizonMemberSpec("tier2-mamba-s6",         "tier2_tactical", "cuda:0", 16.0),
    MultiHorizonMemberSpec("tier2-crossasset-gat",   "tier2_tactical", "cpu",    8.0),
    MultiHorizonMemberSpec("tier2-cryptobert-news",  "tier2_tactical", "cpu",    8.0),
    # Tier 1 Micro (3 models)
    MultiHorizonMemberSpec("tier1-lgbm-micro-1s10s", "tier1_micro",    "cpu",    10.0),
    MultiHorizonMemberSpec("tier1-spot1s-leadlag",   "tier1_micro",    "cpu",    10.0),
    MultiHorizonMemberSpec("tier1-queue-hazard",     "tier1_micro",    "cpu",    8.0),
)

ALL_V4_MEMBER_NAMES: list[str] = [m.member_name for m in MULTI_HORIZON_MEMBER_REGISTRY]
ALL_V4_MEMBERS: list[str] = ALL_V4_MEMBER_NAMES + [
    "fusion-v4-hierarchical",
    "stage5-hierarchical-stacker",
    "stage5-meta-label-sizing",
]

CPU_BOUND_MEMBER_KEYWORDS = (
    "lgbm", "gbm", "hmm", "gat", "cryptobert", "sentiment", "queue-hazard", "spot1s-leadlag",
)
GPU_BOUND_MEMBER_KEYWORDS = (
    "kronos", "patchtst", "tft", "itransformer", "nhits", "mamba", "ssm",
)


def _classify_member_device_type(member_name: str) -> str:
    low = member_name.lower()
    if any(k in low for k in CPU_BOUND_MEMBER_KEYWORDS):
        return "cpu"
    if any(k in low for k in GPU_BOUND_MEMBER_KEYWORDS):
        return "gpu"
    return "any"


class SessionBudgetV4:
    """
    Enforces the 8.5-hour (30,600s) soft-stop threshold so the active model
    always finishes uploading its weights + OOF Parquet to Hugging Face before
    Kaggle's 9-hour / 12-hour hard VM termination.
    """

    def __init__(self, max_hours: float = 8.5, hard_limit_hours: float = 9.0, safety_margin_minutes: float = 15.0) -> None:
        self.start_monotonic = time.monotonic()
        self.soft_stop_sec = float(max_hours * 3600.0)
        self.hard_limit_sec = float(hard_limit_hours * 3600.0)
        self.budget_seconds = self.soft_stop_sec
        self.safety_margin_seconds = float(safety_margin_minutes * 60.0)

    def elapsed_seconds(self) -> float:
        return float(time.monotonic() - self.start_monotonic)

    def remaining_soft_seconds(self) -> float:
        return max(0.0, self.soft_stop_sec - self.elapsed_seconds())

    def remaining_seconds(self) -> float:
        return self.remaining_soft_seconds()

    def should_soft_stop(self) -> bool:
        return self.elapsed_seconds() >= self.soft_stop_sec

    def should_stop(self) -> bool:
        return self.should_soft_stop()

    def check_can_start(self, stage_name: str = "", estimated_duration_seconds: float = 0.0) -> bool:
        return not self.should_soft_stop()


def check_v4_ensemble_status_on_hf(
    model_repo_id: str | None = None,
    token: str | None = None,
) -> dict[str, list[str]]:
    completed: list[str] = []
    pending: list[str] = []
    for spec in MULTI_HORIZON_MEMBER_REGISTRY:
        if is_member_completed_on_hf(spec.member_name, repo_id=model_repo_id, token=token):
            completed.append(spec.member_name)
        else:
            pending.append(spec.member_name)
    fusion_done = is_member_completed_on_hf("fusion-v4-hierarchical", repo_id=model_repo_id, token=token)
    return {
        "completed": completed,
        "pending": pending,
        "fusion_completed": ["fusion-v4-hierarchical"] if fusion_done else [],
    }


def execute_wave_parallel(
    wave_name: str,
    tasks: list[tuple[str, Callable[[str], dict[str, Any]]]],
    output_root: Path,
    budget: Any,
    repo_id: str | None = None,
    token: str | None = None,
    force_retrain: set[str] | list[str] | None = None,
) -> dict[str, str]:
    """
    Executes a wave of model training tasks across Dual T4 GPUs (cuda:0, cuda:1)
    and CPU workers. Routes GPU models to `cuda:0`/`cuda:1` and CPU-bound
    LightGBM/HMM models to `cpu` sequentially so two CPU-bound processes never
    contend for 30GB RAM simultaneously.
    """
    status_map: dict[str, str] = {}
    pending_tasks: list[tuple[str, Callable[[str], dict[str, Any]]]] = []
    raw_force = set(force_retrain or [])
    force_set = raw_force | {canonical_v4_member_name(m) for m in raw_force}

    def _is_soft_stop() -> bool:
        if hasattr(budget, "should_soft_stop"):
            return bool(budget.should_soft_stop())
        if hasattr(budget, "should_stop"):
            return bool(budget.should_stop())
        return False

    for member_name, train_fn in tasks:
        canon_name = canonical_v4_member_name(member_name)
        force_all = "all" in force_set or wave_name in force_set
        in_force_list = member_name in force_set or canon_name in force_set
        target_repo = repo_id or v4_tier_repo_id(canon_name)
        if not force_all and is_member_completed_on_hf(
            canon_name,
            repo_id=target_repo,
            token=token,
            require_v4_peak=in_force_list,
        ):
            log.info("[%s] Skipping %s — already completed on Hugging Face Hub (v4_peak=%s)", wave_name, member_name, in_force_list)
            status_map[member_name] = "skipped_completed_on_hf"
            continue
        if _is_soft_stop():
            log.warning("[%s] Soft-stop budget reached before %s — deferring to next Kaggle session", wave_name, member_name)
            status_map[member_name] = "deferred_soft_stop"
            continue
        pending_tasks.append((member_name, train_fn))

    if not pending_tasks:
        return status_map

    lane_queues: dict[str, list[tuple[str, Callable[[str], dict[str, Any]]]]] = {
        "cuda:0": [],
        "cuda:1": [],
        "cpu": [],
    }
    gpu_lanes = ["cuda:0", "cuda:1"]
    all_lanes = ["cuda:0", "cuda:1", "cpu"]
    gpu_rr = 0
    any_rr = 0

    for m_name, fn in pending_tasks:
        dev_type = _classify_member_device_type(m_name)
        if dev_type == "cpu":
            lane_queues["cpu"].append((m_name, fn))
        elif dev_type == "gpu":
            target_gpu = gpu_lanes[gpu_rr % len(gpu_lanes)]
            gpu_rr += 1
            lane_queues[target_gpu].append((m_name, fn))
        else:
            target_lane = all_lanes[any_rr % len(all_lanes)]
            any_rr += 1
            lane_queues[target_lane].append((m_name, fn))

    def _run_lane_queue(device_str: str, queue_items: list[tuple[str, Callable[[str], dict[str, Any]]]]) -> list[tuple[str, str]]:
        lane_outcomes: list[tuple[str, str]] = []
        for m_name, fn in queue_items:
            if _is_soft_stop():
                lane_outcomes.append((m_name, "deferred_soft_stop"))
                continue
            canon_name = canonical_v4_member_name(m_name)
            member_dir = output_root / m_name
            member_dir.mkdir(parents=True, exist_ok=True)
            log.info("[%s] Starting %s on device %s...", wave_name, m_name, device_str)
            try:
                metrics = fn(device_str)
                metrics_out = dict(metrics or {})
                metrics_out.setdefault("member", m_name)
                metrics_out.setdefault("canonical_member", canon_name)
                metrics_out.setdefault("device", device_str)
                metrics_out.setdefault("status", "completed")
                metrics_out["v4_peak_calibrated"] = True
                (member_dir / "metrics.json").write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")
                target_repo = repo_id or v4_tier_repo_id(canon_name)
                upload_member_checkpoint(m_name, output_root, repo_id=target_repo, token=token)
                lane_outcomes.append((m_name, "completed"))
            except Exception as exc:
                log.exception("[%s] Member %s failed on %s: %s", wave_name, m_name, device_str, exc)
                lane_outcomes.append((m_name, f"failed: {exc}"))
        return lane_outcomes

    active_lanes = {lane: q for lane, q in lane_queues.items() if q}
    with ThreadPoolExecutor(max_workers=max(1, len(active_lanes))) as pool:
        futs = [pool.submit(_run_lane_queue, lane, q) for lane, q in active_lanes.items()]
        for fut in as_completed(futs):
            for m_name, outcome in fut.result():
                status_map[m_name] = outcome

    return status_map


def execute_dual_t4_tier_training_wave(
    tier_name: str,
    member_runners: dict[str, Callable[[str], Any]],
    budget: Any,
    output_root: Path,
    model_repo_id: str | None = None,
    hf_token: str | None = None,
    force_retrain: list[str] | set[str] | None = None,
) -> bool:
    tasks = [(m_name, fn) for m_name, fn in member_runners.items()]
    res = execute_wave_parallel(
        wave_name=tier_name,
        tasks=tasks,
        output_root=output_root,
        budget=budget,
        repo_id=model_repo_id,
        token=hf_token,
        force_retrain=force_retrain,
    )
    return all(v in ("completed", "skipped_completed_on_hf") for v in res.values())
