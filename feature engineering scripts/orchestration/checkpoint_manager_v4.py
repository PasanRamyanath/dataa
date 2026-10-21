"""
feature engineering scripts/orchestration/checkpoint_manager_v4.py
Manages all 15 multi-horizon ensemble members across Tier 4 (Macro), Tier 3 (Meso),
Tier 2 (Tactical), and Tier 1 (Micro Sniper), plus Stage-5 Supervised Hierarchical Fusion.
Preserves 100% compatibility with SessionBudget, poll_and_relaunch.py, and atomic
Hugging Face Hub completion markers (metrics.json uploaded last).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import Callable

from common.hf_model_upload import (
    is_member_completed_on_hf,
    upload_member_checkpoint,
)
from orchestration.checkpoint_manager import SessionBudget

log = logging.getLogger("orchestration.checkpoint_v4")


@dataclass(frozen=True)
class MultiHorizonMemberSpec:
    member_name: str
    tier: str                      # "tier4_macro", "tier3_meso", "tier2_tactical", "tier1_micro"
    preferred_device: str          # "cuda:0", "cuda:1", or "cpu"
    estimated_minutes_t4: float
    requires_oof_parquet: bool = True


MULTI_HORIZON_MEMBER_REGISTRY: tuple[MultiHorizonMemberSpec, ...] = (
    MultiHorizonMemberSpec("tier4-hmm-regime",       "tier4_macro",    "cpu",    4.0),
    MultiHorizonMemberSpec("tier4-patchtst-ffd",     "tier4_macro",    "cuda:0", 14.0),
    MultiHorizonMemberSpec("tier4-tft-funding",      "tier4_macro",    "cuda:0", 16.0),
    MultiHorizonMemberSpec("tier4-kronos-macro",     "tier4_macro",    "cuda:1", 22.0),
    MultiHorizonMemberSpec("tier3-lgbm-positioning", "tier3_meso",     "cpu",    8.0),
    MultiHorizonMemberSpec("tier3-itransformer-32",  "tier3_meso",     "cuda:0", 25.0),
    MultiHorizonMemberSpec("tier3-nhits-multirate",  "tier3_meso",     "cuda:0", 20.0),
    MultiHorizonMemberSpec("tier3-kronos-meso",      "tier3_meso",     "cuda:1", 28.0),
    MultiHorizonMemberSpec("tier2-lgbm-bookdepth",   "tier2_tactical", "cpu",    10.0),
    MultiHorizonMemberSpec("tier2-mamba-s6",         "tier2_tactical", "cuda:0", 26.0),
    MultiHorizonMemberSpec("tier2-crossasset-gat",   "tier2_tactical", "cuda:1", 18.0),
    MultiHorizonMemberSpec("tier2-cryptobert-news",  "tier2_tactical", "cuda:1", 12.0),
    MultiHorizonMemberSpec("tier1-lgbm-micro-1s10s", "tier1_micro",    "cpu",    12.0),
    MultiHorizonMemberSpec("tier1-spot1s-leadlag",   "tier1_micro",    "cuda:0", 10.0),
    MultiHorizonMemberSpec("tier1-queue-hazard",     "tier1_micro",    "cpu",    8.0),
)

ALL_V4_MEMBER_NAMES: list[str] = [m.member_name for m in MULTI_HORIZON_MEMBER_REGISTRY]


def check_v4_ensemble_status_on_hf(
    model_repo_id: str,
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


def execute_dual_t4_tier_training_wave(
    tier_name: str,
    member_runners: dict[str, Callable[[str], None]],
    budget: SessionBudget,
    output_root: Path,
    model_repo_id: str,
    hf_token: str | None = None,
    force_retrain: list[str] | set[str] | None = None,
) -> bool:
    tier_specs = [s for s in MULTI_HORIZON_MEMBER_REGISTRY if s.tier == tier_name]
    force_set = set(force_retrain or [])
    retrain_all = "all" in force_set or tier_name in force_set
    pending_specs = [
        s for s in tier_specs
        if retrain_all or s.member_name in force_set or not is_member_completed_on_hf(s.member_name, repo_id=model_repo_id, token=hf_token)
    ]
    if not pending_specs:
        log.info("Tier %s already 100%% complete on Hugging Face Hub.", tier_name)
        return True

    lanes: dict[str, list[MultiHorizonMemberSpec]] = {"cuda:0": [], "cuda:1": [], "cpu": []}
    for spec in pending_specs:
        lanes.setdefault(spec.preferred_device, []).append(spec)

    def _run_device_lane(device_str: str, specs: list[MultiHorizonMemberSpec]) -> bool:
        for spec in specs:
            if not budget.check_can_start(
                stage_name=spec.member_name,
                estimated_duration_seconds=spec.estimated_minutes_t4 * 60.0,
            ):
                log.warning(
                    "SessionBudget remaining (%.1f min) insufficient for %s (est %.1f min). Yielding to next relay session.",
                    budget.remaining_seconds() / 60.0,
                    spec.member_name,
                    spec.estimated_minutes_t4,
                )
                return False

            runner_fn = member_runners.get(spec.member_name)
            if runner_fn is None:
                raise KeyError(f"Missing training runner for {spec.member_name}")

            t0 = time.time()
            log.info("Starting %s on device %s...", spec.member_name, device_str)
            runner_fn(device_str)
            elapsed_m = (time.time() - t0) / 60.0
            log.info("Finished %s in %.2f min. Uploading atomically to HF (%s)...", spec.member_name, elapsed_m, model_repo_id)

            upload_member_checkpoint(
                member_name=spec.member_name,
                output_root=output_root,
                repo_id=model_repo_id,
                token=hf_token,
            )
        return True

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(_run_device_lane, dev, sp_list): dev
            for dev, sp_list in lanes.items()
            if sp_list
        }
        all_ok = True
        for fut in as_completed(futures):
            if not fut.result():
                all_ok = False
    return all_ok
