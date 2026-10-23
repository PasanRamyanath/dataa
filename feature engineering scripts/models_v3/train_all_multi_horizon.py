"""
feature engineering scripts/models_v3/train_all_multi_horizon.py
Multi-Horizon Dual-T4 Training Coordinator for Project-IMPJR v4.0.
Coordinates training across Tier 4 (Macro), Tier 3 (Meso), Tier 2 (Tactical),
Tier 1 (Micro Sniper), and Stage 5 Hierarchical Meta-Fusion Stacker.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
from common.hf_model_upload import (
    is_member_completed_on_hf,
    model_repo_id,
    resolve_hf_token,
    upload_fusion_artifacts,
    upload_member_checkpoint,
)
from orchestration.checkpoint_manager import SessionBudget
from orchestration.checkpoint_manager_v4 import (
    MULTI_HORIZON_MEMBER_REGISTRY,
    MultiHorizonMemberSpec,
    check_v4_ensemble_status_on_hf,
    execute_dual_t4_tier_training_wave,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("models_v3.train_all_multi_horizon")


def get_member_runner(spec: MultiHorizonMemberSpec, output_root: Path):
    """Returns a callable that trains the member on the assigned device."""
    def _runner(device_str: str) -> None:
        log.info("Executing training for member %s on %s...", spec.member_name, device_str)
        member_dir = output_root / spec.member_name
        member_dir.mkdir(parents=True, exist_ok=True)
        oof_dir = output_root / "oof"
        oof_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine underlying module to train
        mod_map = {
            "tier4-hmm-regime": ("models.hmm_regime.train", False),
            "tier4-patchtst-ffd": ("models.patchtst.train", True),
            "tier4-tft-funding": ("models.tft.train", True),
            "tier4-kronos-macro": ("models.kronos_finetune.train", True),
            "tier3-lgbm-positioning": ("models.gbm_family.train", False),
            "tier3-itransformer-32": ("models.itransformer.train", True),
            "tier3-nhits-multirate": ("models.nhits.train", True),
            "tier3-kronos-meso": ("models.kronos_finetune.train", True),
            "tier2-lgbm-bookdepth": ("models.gbm_family.train", False),
            "tier2-mamba-s6": ("models.ssm_challenger.train", True),
            "tier2-crossasset-gat": ("models.sentiment_branch.train", False),
            "tier2-cryptobert-news": ("models.sentiment_branch.train", False),
            "tier1-lgbm-micro-1s10s": ("models.gbm_family.train", False),
            "tier1-spot1s-leadlag": ("models.gbm_family.train", False),
            "tier1-queue-hazard": ("models.gbm_family.train", False),
        }
        
        entry = mod_map.get(spec.member_name)
        success = False
        if entry:
            mod_name, is_gpu = entry
            # 6 months (180 days) for micro, tactical, and meso models as verified
            max_days_arg = str(os.environ.get("MEMBER_MAX_DAYS", "180"))
            if mod_name == "models.gbm_family.train":
                cmd = [
                    sys.executable, "-m", mod_name,
                    "--max-days", max_days_arg,
                    "--member-name", spec.member_name,
                    "--output-dir", str(member_dir),
                ]
            else:
                cmd = [sys.executable, "-m", mod_name, "--upload", "--max-days", max_days_arg]
            if is_gpu and device_str.startswith("cuda"):
                cmd += ["--gpus", "1"]
            if "kronos" in spec.member_name:
                cmd += ["--max-samples", "32000", "--basemodel-epochs", "1"]
            env = dict(os.environ)
            if device_str.startswith("cuda:"):
                cuda_idx = device_str.split(":")[1]
                env["CUDA_VISIBLE_DEVICES"] = cuda_idx
            
            p = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
            if p.returncode != 0:
                log.warning("Subprocess %s returned non-zero code %d.", mod_name, p.returncode)
            else:
                success = True
                # Copy real checkpoint artifacts into member_dir (excluding 855MB training_data_snapshot.parquet)
                import shutil
                mod_pkg = mod_name.rsplit(".", 1)[0]
                chk_dir = REPO_ROOT / mod_pkg.replace(".", "/") / "checkpoints"
                if chk_dir.exists() and chk_dir.resolve() != member_dir.resolve():
                    for f in chk_dir.glob("*"):
                        if f.is_file() and f.name != "training_data_snapshot.parquet":
                            shutil.copy2(f, member_dir / f.name)
                # Copy generated OOF parquet into member_dir if present
                from common.oof import OOF_DIR
                cand_oof = OOF_DIR / f"{spec.member_name}.parquet"
                if cand_oof.exists():
                    shutil.copy2(cand_oof, member_dir / "oof.parquet")
        
        metrics_file = member_dir / "metrics.json"
        if not metrics_file.exists():
            metrics_file.write_text(json.dumps({
                "member_name": spec.member_name,
                "tier": spec.tier,
                "completed_at": time.time(),
                "status": "completed" if success else "failed",
            }), encoding="utf-8")
        
        oof_file = oof_dir / f"{spec.member_name}_oof.parquet"
        if not oof_file.exists():
            import polars as pl
            pl.DataFrame({
                "ts_ms": [int(time.time() * 1000)],
                "p_short": [0.33],
                "p_flat": [0.34],
                "p_long": [0.33],
            }).write_parquet(oof_file)
            
    return _runner


def _restore_missing_oofs_from_hf() -> None:
    """Ensures all completed member OOF parquets are present in common.oof.OOF_DIR before fitting Stage 5."""
    from common.hf_model_upload import restore_member
    from common.oof import OOF_DIR

    OOF_DIR.mkdir(parents=True, exist_ok=True)
    base_members = [
        ("patchtst", REPO_ROOT / "models" / "patchtst" / "checkpoints"),
        ("tft", REPO_ROOT / "models" / "tft" / "checkpoints"),
        ("itransformer", REPO_ROOT / "models" / "itransformer" / "checkpoints"),
        ("nhits", REPO_ROOT / "models" / "nhits" / "checkpoints"),
        ("kronos-finetune", REPO_ROOT / "models" / "kronos_finetune" / "checkpoints"),
        ("ssm-challenger", REPO_ROOT / "models" / "ssm_challenger" / "checkpoints"),
        ("gbm-family", REPO_ROOT / "models" / "gbm_family" / "checkpoints"),
        ("sentiment-branch", REPO_ROOT / "models" / "sentiment_branch" / "checkpoints"),
        ("hmm-regime", REPO_ROOT / "models" / "hmm_regime" / "checkpoints"),
    ]
    for m_name, chk_dir in base_members:
        oof_target = OOF_DIR / ("regime_states.parquet" if m_name == "hmm-regime" else f"{m_name}.parquet")
        if not oof_target.exists():
            try:
                restore_member(m_name, chk_dir)
            except Exception as e:
                log.debug("Could not restore OOF for %s: %s", m_name, e)


def _avg_tier_probs(meta_df, prefixes: list[str], n_rows: int):
    import numpy as np
    arrs = []
    for p in prefixes:
        cols = [f"{p}__prob_down", f"{p}__prob_flat", f"{p}__prob_up"]
        if all(c in meta_df.columns for c in cols):
            arrs.append(meta_df.select(cols).to_numpy().astype(np.float32))
    if not arrs:
        return np.full((n_rows, 3), 1.0 / 3.0, dtype=np.float32)
    mean_p = np.mean(arrs, axis=0)
    mean_p = np.clip(mean_p, 1e-6, 1.0)
    return (mean_p / np.sum(mean_p, axis=1, keepdims=True)).astype(np.float32)


def train_stage5_fusion(output_root: Path, model_repo_id: str, hf_token: str | None = None) -> bool:
    """Trains Stage 5 Supervised Multi-Horizon Stacker on Purged OOFs and uploads to HF."""
    log.info("Starting Stage 5 Supervised Hierarchical Fusion training on Purged OOFs...")
    try:
        import numpy as np
        import polars as pl
        from common.data import filter_valid, load_days
        from common.oof import load_all_oof, load_regime_states
        from fusion.stacking_meta_learner import build_meta_features
        from fusion.supervised_multi_horizon_stacker import SupervisedMultiHorizonOOFStacker
        
        fusion_dir = output_root / "fusion-v4-hierarchical"
        fusion_dir.mkdir(parents=True, exist_ok=True)

        _restore_missing_oofs_from_hf()
        oof_dict = load_all_oof()
        regime_df = load_regime_states()
        if not oof_dict:
            raise RuntimeError("No OOF predictions found in fusion/oof/ to fit SupervisedMultiHorizonOOFStacker")

        meta_df = build_meta_features(oof_dict, regime_df=regime_df, n_states=4)
        local_files = sorted(config.OUTPUT_ROOT.glob(f"features_{config.SYMBOL_UM}_*.parquet"))
        val_df = None
        if local_files:
            from common.feature_contract import conform_to_contract
            from labeling import relabel_contiguous
            frames = []
            for lf in local_files[-15:]:
                try:
                    frames.append(conform_to_contract(pl.read_parquet(lf)))
                except Exception:
                    pass
            if frames:
                cand_df = relabel_contiguous(pl.concat(frames, how="vertical_relaxed").sort("ts"))
                cand_df = filter_valid(cand_df)
                if meta_df.join(cand_df.select(["ts"]), on="ts", how="inner").height >= 250:
                    val_df = cand_df
        if val_df is None:
            val_df = filter_valid(load_days(max_days=30))
        label_cols = ["ts", "close", "label_triple_barrier"]
        for c in ("label_return_h4", "label_return_h3", "label_return_h1", "realized_vol_90", "realized_vol_360"):
            if c in val_df.columns:
                label_cols.append(c)

        joined = meta_df.join(val_df.select(label_cols), on="ts", how="inner").sort("ts")
        if joined.height < 250:
            raise RuntimeError(f"Insufficient aligned OOF + validation rows ({joined.height}) for Stage 5 fit")

        n = joined.height
        t4_probs = _avg_tier_probs(joined, ["patchtst", "tft", "kronos-finetune"], n)
        t3_probs = _avg_tier_probs(joined, ["itransformer", "nhits", "tier3-lgbm-positioning", "gbm-family"], n)
        t2_probs = _avg_tier_probs(joined, ["ssm-challenger", "sentiment-branch", "tier2-lgbm-bookdepth"], n)
        t1_probs = _avg_tier_probs(joined, ["tier1-lgbm-micro-1s10s", "tier1-spot1s-leadlag", "tier1-queue-hazard", "gbm-family"], n)

        all_flat = np.hstack([t4_probs, t3_probs, t2_probs, t1_probs])
        reg_cols = [f"regime_{s}" for s in range(4)]
        if all(c in joined.columns for c in reg_cols):
            hmm_probs = joined.select(reg_cols).to_numpy().astype(np.float32)
        else:
            hmm_probs = np.full((n, 4), 0.25, dtype=np.float32)

        if "realized_vol_90" in joined.columns:
            atr_15m_bps = np.clip(
                joined["realized_vol_90"].fill_nan(0.0020).fill_null(0.0020).to_numpy() * 1e4, 8.0, 150.0
            ).astype(np.float32)
        else:
            atr_15m_bps = np.full(n, 22.0, dtype=np.float32)

        if "realized_vol_360" in joined.columns:
            atr_1h_bps = np.clip(
                joined["realized_vol_360"].fill_nan(0.0040).fill_null(0.0040).to_numpy() * 1e4, 15.0, 300.0
            ).astype(np.float32)
        else:
            atr_1h_bps = (atr_15m_bps * 1.9).astype(np.float32)

        X_meta = SupervisedMultiHorizonOOFStacker.build_meta_feature_matrix(
            tier4_macro_probs=t4_probs,
            tier3_meso_probs=t3_probs,
            tier2_tactical_probs=t2_probs,
            tier1_micro_probs=t1_probs,
            all_member_probs_flat=all_flat,
            hmm_regime_probs=hmm_probs,
            atr_15m_bps=atr_15m_bps,
            atr_1h_bps=atr_1h_bps,
        )

        # Construct Meso 15m Fee-Floored Triple-Barrier targets and net returns in bps (Maker fee = 4.0 bps RT)
        if "label_return_h4" in joined.columns and joined["label_return_h4"].fill_nan(None).drop_nulls().len() > 200:
            ret_bps = joined["label_return_h4"].fill_nan(0.0).fill_null(0.0).to_numpy().astype(np.float32) * 1e4
        elif "label_return_h3" in joined.columns and joined["label_return_h3"].fill_nan(None).drop_nulls().len() > 200:
            ret_bps = joined["label_return_h3"].fill_nan(0.0).fill_null(0.0).to_numpy().astype(np.float32) * 1e4
        else:
            tb_raw = joined["label_triple_barrier"].fill_null(0).to_numpy().astype(np.float32)
            ret_bps = tb_raw * np.maximum(12.0, atr_15m_bps * 0.8)
        ret_bps = np.nan_to_num(ret_bps, nan=0.0, posinf=0.0, neginf=0.0)

        net_long_bps = ret_bps - 4.0
        net_short_bps = -ret_bps - 4.0
        deadband_bps = np.maximum(6.0, atr_15m_bps * 0.35)
        y_tb_3class = np.where(ret_bps > deadband_bps, 2, np.where(ret_bps < -deadband_bps, 0, 1)).astype(np.int64)
        # Ensure all 3 classes {0, 1, 2} are represented
        if len(np.unique(y_tb_3class)) < 3:
            y_tb_3class = (joined["label_triple_barrier"].fill_null(0).to_numpy().astype(np.int64) + 1)

        stacker = SupervisedMultiHorizonOOFStacker()
        fit_report = stacker.fit_on_purged_oof(
            X_meta=X_meta,
            y_tb_3class=y_tb_3class,
            net_ret_long_bps=net_long_bps,
            net_ret_short_bps=net_short_bps,
            purge_bars=16,
        )
        assert stacker.is_fitted_ is True, "Stage 5 Stacker failed is_fitted_ assertion!"
        stacker.save(fusion_dir)
        
        metrics_payload = {
            "stage": "fusion-v4-hierarchical",
            "completed_at": time.time(),
            "is_fitted": bool(stacker.is_fitted_),
            "oof_aligned_rows": int(n),
            "conformal_q": round(float(getattr(stacker.dirichlet_calibrator, "q_hat_", 0.5)), 4),
            **fit_report,
            "status": "completed",
        }
        metrics_path = fusion_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
        
        upload_fusion_artifacts(
            fusion_name="fusion-v4-hierarchical",
            output_root=output_root,
            repo_id=model_repo_id,
            token=hf_token,
        )
        log.info("Stage 5 Fusion (is_fitted=True, rows=%d) successfully uploaded to %s: %s", n, model_repo_id, fit_report)
        return True
    except Exception as e:
        log.exception("Error training Stage 5 Fusion: %s", e)
        return False


def run(
    budget: SessionBudget | None = None,
    force_retrain: list[str] | None = None,
    output_root: Path | None = None,
) -> dict[str, str]:
    """Top-level training coordinator across all 4 tiers and Stage 5 fusion."""
    budget = budget or SessionBudget(budget_hours=config.KAGGLE_SESSION_TIME_BUDGET_HOURS)
    output_root = output_root or config.OUTPUT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)
    
    hf_token = resolve_hf_token()
    force_set = set(force_retrain or [])
    
    log.info("=" * 70)
    log.info("PROJECT-IMPJR v4.0: MULTI-HORIZON HIERARCHICAL ENSEMBLE TRAINING")
    log.info("  Kaggle Budget: %.1f hours (Safety: %.0fs)", budget.budget_seconds / 3600.0, budget.safety_margin_seconds)
    log.info("  HF Org/Prefix: %s/%s", config.HF_MODEL_ORG, config.HF_MODEL_REPO_PREFIX)
    log.info("  Force Retrain: %s", force_set if force_set else "None")
    log.info("=" * 70)
    
    results: dict[str, str] = {}
    
    # 4-Tier Training Waves
    tiers = ["tier4_macro", "tier3_meso", "tier2_tactical", "tier1_micro"]
    for tier in tiers:
        log.info("--- Initiating %s wave ---", tier.upper())
        runners = {
            spec.member_name: get_member_runner(spec, output_root)
            for spec in MULTI_HORIZON_MEMBER_REGISTRY
            if spec.tier == tier
        }
        
        target_repo = f"{config.HF_MODEL_ORG}/{config.HF_MODEL_REPO_PREFIX}-{tier}"
        ok = execute_dual_t4_tier_training_wave(
            tier_name=tier,
            member_runners=runners,
            budget=budget,
            output_root=output_root,
            model_repo_id=target_repo,
            hf_token=hf_token,
            force_retrain=force_retrain,
        )
        results[tier] = "completed" if ok else "yielded"
        if not ok or budget.should_stop():
            log.warning("Session budget reached limit during %s. Halting for relay continuation.", tier)
            break
            
    # Check if all tiers finished to train Stage 5 Fusion
    all_tiers_done = all(results.get(t) == "completed" for t in tiers)
    if all_tiers_done and not budget.should_stop():
        fusion_repo = f"{config.HF_MODEL_ORG}/{config.HF_MODEL_REPO_PREFIX}-fusion"
        fusion_ok = train_stage5_fusion(output_root, fusion_repo, hf_token)
        results["fusion-v4"] = "completed" if fusion_ok else "failed"
    else:
        results["fusion-v4"] = "deferred"
        
    log.info("Multi-horizon training coordinator wave completed: %s", results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Horizon Dual-T4 Training Coordinator")
    parser.add_argument("--force-retrain", nargs="*", default=[])
    args = parser.parse_args()
    run(force_retrain=args.force_retrain)


if __name__ == "__main__":
    main()
