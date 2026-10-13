"""
Kronos (arxiv.org/abs/2508.02739) finetune ensemble member.

Kronos is NOT pip-installable — it ships only as a GitHub repo
(github.com/shiyu-coder/Kronos, confirmed via web research 2026-09-12) with
its tokenizer/predictor classes imported directly from that repo's `model/`
package, and its two-stage finetuning (tokenizer, then predictor) driven by
`finetune_csv/train_sequential.py` + a YAML config — NOT the Qlib-pickle
`finetune/` path, which is oriented at daily-bar multi-instrument equities
and a poor fit for our single-symbol 10s-bar crypto series. This script
therefore:

  1. Clones the upstream repo into vendor/Kronos if not already present
     (never vendored/copied into this project directly — see ensure_repo).
  2. Downloads the pretrained Kronos-Tokenizer-base + Kronos-base checkpoints
     from Hugging Face (NeoQuasar/Kronos-Tokenizer-base, NeoQuasar/Kronos-base).
  3. Exports OUR OWN purged/embargoed TRAIN split (common.cv, never the val
     split) to the exact CSV schema finetune_csv expects: timestamps, open,
     high, low, close, volume, amount — using notional_volume as `amount`
     (quote-asset volume), matching Binance's own quoteVolume convention.
  4. Writes a config YAML mirroring finetune_csv's reference schema
     (configs/config_ali09988_candle-5min.yaml) and runs
     finetune_csv/train_sequential.py as a subprocess against OUR CSV — the
     repo's own train_ratio/val_ratio split happens only WITHIN our
     already-purged train portion, so it never touches held-out data.
  5. Independently evaluates the finetuned checkpoint on OUR held-out val
     split via KronosPredictor.predict (never the repo's own internal val
     metric), so this member's reported metrics are directly comparable to
     every other member's (same purged/embargoed val window, same
     directional-accuracy definition as common/neuralforecast_harness.py).

max_context for Kronos-small/base is 512 (confirmed via web research), below
config.LOOKBACK_PERIODS=720 — lookback_window is therefore capped at 512
here, not 720; every other sequence member keeps the full 720-bar lookback.

Usage:
    python -m models.kronos_finetune.train --val-days 7 --upload
    python -m models.kronos_finetune.train --skip-finetune --upload   # zero-shot frozen-checkpoint eval only
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import yaml

import config
from common.cv import purged_walk_forward_splits
from common.data import filter_valid, load_days

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("models.kronos_finetune.train")

MEMBER_NAME = "kronos-finetune"
MODULE_DIR = Path(__file__).resolve().parent
VENDOR_DIR = MODULE_DIR / "vendor" / "Kronos"
PRETRAINED_DIR = MODULE_DIR / "pretrained"
DATA_DIR = MODULE_DIR / "data"
FINETUNED_DIR = MODULE_DIR / "finetuned"
CHECKPOINT_DIR = MODULE_DIR / "checkpoints"

KRONOS_REPO_URL = "https://github.com/shiyu-coder/Kronos.git"
TOKENIZER_REPO_ID = "NeoQuasar/Kronos-Tokenizer-base"
PREDICTOR_REPO_ID = "NeoQuasar/Kronos-base"
MAX_CONTEXT = 512  # Kronos-small/base's documented context cap; see module docstring
EXP_NAME = f"{config.SYMBOL_UM}_{config.PRIMARY_INTERVAL}"


def ensure_repo() -> Path:
    if not (VENDOR_DIR / "finetune_csv").exists():
        VENDOR_DIR.parent.mkdir(parents=True, exist_ok=True)
        log.info("Cloning %s into %s", KRONOS_REPO_URL, VENDOR_DIR)
        subprocess.run(["git", "clone", "--depth", "1", KRONOS_REPO_URL, str(VENDOR_DIR)], check=True)
    return VENDOR_DIR


def ensure_pretrained_checkpoints() -> tuple[Path, Path]:
    from huggingface_hub import snapshot_download

    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer_path = Path(snapshot_download(repo_id=TOKENIZER_REPO_ID, local_dir=str(PRETRAINED_DIR / "Kronos-Tokenizer-base")))
    predictor_path = Path(snapshot_download(repo_id=PREDICTOR_REPO_ID, local_dir=str(PRETRAINED_DIR / "Kronos-base")))
    return tokenizer_path, predictor_path


def export_csv(train_df: pl.DataFrame) -> Path:
    """finetune_csv's expected schema: timestamps, open, high, low, close,
    volume, amount — `amount` is quote-asset (notional) volume, matching
    Binance's own quoteVolume convention; we already compute this as
    notional_volume (feature_engine/price_volume.py)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = train_df.select(
        pl.col("ts").alias("timestamps"),
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Float64),
        pl.col("notional_volume").cast(pl.Float64).alias("amount"),
    )
    csv_path = DATA_DIR / f"{EXP_NAME}_train.csv"
    out.write_csv(csv_path)
    return csv_path


def write_config(csv_path: Path, tokenizer_path: Path, predictor_path: Path,
                  lookback_window: int, predict_window: int, tokenizer_epochs: int,
                  basemodel_epochs: int, batch_size: int, resume: bool = False) -> Path:
    """Mirrors finetune_csv/configs/config_ali09988_candle-5min.yaml's exact
    key structure (confirmed via direct source read of
    github.com/shiyu-coder/Kronos@master finetune_csv/config_loader.py,
    2026-09-12) — any key mismatch here would fail loudly inside the
    vendored repo's own config_loader.py rather than silently, since we
    never modify that file.

    `distributed.use_ddp`/`backend` are written here for schema completeness
    but are NOT what actually activates multi-GPU training in
    train_sequential.py — confirmed via direct source read that
    SequentialTrainer._setup_distributed() gates purely on the WORLD_SIZE
    env var (set by `torchrun`/`torch.distributed.run`), not on this config
    section. Real multi-GPU activation is run_finetune()'s job (launching
    via torchrun when gpus>1), not this function's.

    `resume`, when True, sets `skip_existing=True`: confirmed via direct
    source read that Kronos upstream has no epoch-level resume (no
    ckpt_path/optimizer-state persistence anywhere in finetune_csv) — the
    only resume primitive it exposes is this binary check against each
    stage's final best_model directory (SequentialTrainer._check_existing_models),
    so a rerun with resume=True skips a stage entirely if its best_model
    already exists (e.g. tokenizer finished before a Kaggle session was
    interrupted) rather than retraining it from scratch, but cannot resume
    a stage that was interrupted mid-epoch."""
    cfg = {
        "data": {
            "data_path": str(csv_path),
            "lookback_window": lookback_window,
            "predict_window": predict_window,
            "max_context": MAX_CONTEXT,
            "clip": 5.0,
            "train_ratio": 0.9,
            "val_ratio": 0.1,
            "test_ratio": 0.0,
        },
        "training": {
            "tokenizer_epochs": tokenizer_epochs,
            "basemodel_epochs": basemodel_epochs,
            "batch_size": batch_size,
            "log_interval": 50,
            "num_workers": 2,
            "seed": 42,
            "tokenizer_learning_rate": 2e-4,
            "predictor_learning_rate": 1e-6,
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "adam_weight_decay": 0.1,
            "accumulation_steps": 1,
        },
        "model_paths": {
            "pretrained_tokenizer": str(tokenizer_path),
            "pretrained_predictor": str(predictor_path),
            "exp_name": EXP_NAME,
            "base_path": str(FINETUNED_DIR),
            "base_save_path": "",
            "finetuned_tokenizer": "",
            "tokenizer_save_name": "tokenizer",
            "basemodel_save_name": "basemodel",
        },
        "experiment": {
            "name": f"kronos_{EXP_NAME}",
            "description": "Project-IMPJR ensemble member finetune",
            "use_comet": False,
            "train_tokenizer": True,
            "train_basemodel": True,
            "skip_existing": resume,
        },
        "device": {"use_cuda": True, "device_id": 0},
        # See docstring: not read by _setup_distributed() (which gates on
        # WORLD_SIZE instead) — kept for schema completeness only.
        "distributed": {"use_ddp": False, "backend": "nccl"},
    }
    config_path = DATA_DIR / f"{EXP_NAME}_config.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return config_path


def run_finetune(vendor_dir: Path, config_path: Path, gpus: int | None = None) -> None:
    """Launches train_sequential.py. When >1 GPU is available/requested,
    launches via `python -m torch.distributed.run` (torchrun's underlying
    module — invoking it this way avoids depending on a `torchrun` console
    script being on PATH) so WORLD_SIZE/RANK/LOCAL_RANK are set for
    SequentialTrainer._setup_distributed() to pick up, confirmed via direct
    source read to be the actual (only) multi-GPU activation path — the
    config YAML's distributed.use_ddp key is not read by that method."""
    import torch

    if gpus is None:
        gpus = torch.cuda.device_count()

    if gpus > 1:
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            "--standalone", "--nproc_per_node", str(gpus),
            "train_sequential.py", "--config", str(config_path),
        ]
    else:
        cmd = [sys.executable, "train_sequential.py", "--config", str(config_path)]
    log.info("Running: %s (cwd=%s)", " ".join(cmd), vendor_dir / "finetune_csv")
    subprocess.run(cmd, cwd=str(vendor_dir / "finetune_csv"), check=True)


def _load_predictor(vendor_dir: Path, tokenizer_path: Path, predictor_path: Path):
    sys.path.insert(0, str(vendor_dir))
    from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: PLC0415 (vendored repo import, must happen after sys.path insert)

    tokenizer = KronosTokenizer.from_pretrained(str(tokenizer_path))
    model = Kronos.from_pretrained(str(predictor_path))
    return KronosPredictor(model, tokenizer, max_context=MAX_CONTEXT)


def evaluate_on_val(
    predictor, val_df: pl.DataFrame, lookback_window: int, predict_window: int
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Independent held-out evaluation, NOT the repo's own internal val
    metric — walks non-overlapping (context, horizon) windows across
    val_df and scores directional accuracy + MAE the same way
    common/neuralforecast_harness.py does, so this member is directly
    comparable to tft/patchtst/itransformer/nhits.

    Also returns, per window, the horizon-end timestamp and the predicted
    return relative to the last known context close — the OOF signal
    fusion.stacking_meta_learner consumes via
    common.oof.regression_to_direction_proba, same convention as every
    other regression-only member."""
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    n = val_df.height
    step = lookback_window + predict_window
    maes, correct_dir, total = [], 0, 0
    window_ts, window_pred_return = [], []

    for start in range(0, n - step, step):
        ctx = val_df.slice(start, lookback_window)
        fut = val_df.slice(start + lookback_window, predict_window)
        try:
            x_df = ctx.select(ohlcv_cols).to_pandas()
            x_ts = ctx["ts"].to_pandas()
            y_ts = fut["ts"].to_pandas()
        except Exception:
            import pandas as pd
            x_df = pd.DataFrame(ctx.select(ohlcv_cols).to_dict(as_series=False))
            x_ts = pd.Series(ctx["ts"].to_list(), name="ts")
            y_ts = pd.Series(fut["ts"].to_list(), name="ts")

        pred_df = predictor.predict(df=x_df, x_timestamp=x_ts, y_timestamp=y_ts, pred_len=predict_window, T=1.0, top_p=0.9, sample_count=1)

        y_true = fut["close"].to_numpy()
        y_pred = pred_df["close"].to_numpy()
        last_known = ctx["close"].to_numpy()[-1]

        maes.append(float(np.mean(np.abs(y_true - y_pred))))
        correct_dir += int(np.sign(y_true[-1] - last_known) == np.sign(y_pred[-1] - last_known))
        total += 1

        window_ts.append(fut["ts"].to_numpy()[-1])
        window_pred_return.append(float((y_pred[-1] - last_known) / last_known) if last_known else 0.0)

    metrics = {
        "mae": float(np.mean(maes)) if maes else None,
        "directional_accuracy": correct_dir / total if total else None,
        "n_windows": total,
    }
    return metrics, np.array(window_ts), np.array(window_pred_return, dtype=np.float32)


def run(val_days: int, max_days: int | None, lookback_window: int, predict_window: int,
        tokenizer_epochs: int, basemodel_epochs: int, batch_size: int,
        skip_finetune: bool, upload: bool, gpus: int | None = None, resume: bool = False) -> dict:
    df = load_days(max_days=max_days)
    val_size = val_days * config.BARS_PER_DAY
    split = purged_walk_forward_splits(df, n_folds=1, val_size=val_size)[0]

    vendor_dir = ensure_repo()
    tokenizer_path, predictor_path = ensure_pretrained_checkpoints()

    csv_path = export_csv(split.train)
    config_path = write_config(csv_path, tokenizer_path, predictor_path, lookback_window,
                                predict_window, tokenizer_epochs, basemodel_epochs, batch_size,
                                resume=resume)

    finetuned_tokenizer_path = tokenizer_path
    finetuned_predictor_path = predictor_path
    if not skip_finetune:
        run_finetune(vendor_dir, config_path, gpus=gpus)
        finetuned_tokenizer_path = FINETUNED_DIR / EXP_NAME / "tokenizer" / "best_model"
        finetuned_predictor_path = FINETUNED_DIR / EXP_NAME / "basemodel" / "best_model"

    predictor = _load_predictor(vendor_dir, finetuned_tokenizer_path, finetuned_predictor_path)
    metrics, oof_ts, oof_pred_return = evaluate_on_val(predictor, split.val, lookback_window, predict_window)
    metrics["skip_finetune"] = skip_finetune
    log.info("Kronos eval: mae=%s directional_accuracy=%s over %d windows",
              metrics["mae"], metrics["directional_accuracy"], metrics["n_windows"])

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    (CHECKPOINT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if len(oof_ts) > 0:
        from common.oof import regression_to_direction_proba, save_oof

        direction_proba = regression_to_direction_proba(oof_pred_return)
        save_oof(MEMBER_NAME, oof_ts, direction_proba)

    if upload:
        from common.hf_model_upload import upload_best_checkpoint, upload_training_data_snapshot

        checkpoint_files = [CHECKPOINT_DIR / "metrics.json"]
        if not skip_finetune:
            checkpoint_files += list(finetuned_tokenizer_path.glob("*")) + list(finetuned_predictor_path.glob("*"))
        repo_id = upload_best_checkpoint(MEMBER_NAME, checkpoint_files, metrics=metrics)
        data_repo_id = upload_training_data_snapshot(MEMBER_NAME, [csv_path])
        log.info("Uploaded checkpoint to %s and training data to %s", repo_id, data_repo_id)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-days", type=int, default=config.VAL_DAYS)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--lookback-window", type=int, default=MAX_CONTEXT)
    parser.add_argument("--predict-window", type=int, default=48)
    parser.add_argument("--tokenizer-epochs", type=int, default=30)
    parser.add_argument("--basemodel-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--skip-finetune", action="store_true", help="zero-shot eval with the frozen pretrained checkpoint only")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--gpus", type=int, default=None,
                         help="GPU count for finetuning; default auto-detects via torch.cuda.device_count(). "
                              ">1 launches via torch.distributed.run (torchrun)")
    parser.add_argument("--resume", action="store_true",
                         help="skip a stage (tokenizer/basemodel) entirely if its best_model already "
                              "exists from a prior interrupted run — Kronos upstream has no mid-epoch "
                              "resume, only this binary per-stage skip (see write_config docstring)")
    args = parser.parse_args()
    run(args.val_days, args.max_days, args.lookback_window, args.predict_window,
        args.tokenizer_epochs, args.basemodel_epochs, args.batch_size, args.skip_finetune, args.upload,
        args.gpus, args.resume)


if __name__ == "__main__":
    main()
