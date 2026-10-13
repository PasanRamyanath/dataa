"""
SSM/Mamba "challenger" ensemble member — see models/ssm_challenger/model.py
for the architecture (real mamba-ssm CUDA block when available, GRU
fallback otherwise). Its role in the plan is to challenge the transformer-
family members (tft/patchtst/itransformer) with a linear-complexity
alternative over the same LOOKBACK_PERIODS window; fusion decides how much
weight it gets based on held-out performance, same as every other member.

Consumes fixed-length lookback windows over the full feature set, sliced
lazily from one shared feature matrix (common.windowing.LazyWindowDataset —
never materialized, so a full-history run fits in RAM) over the UNFILTERED
day concat: windows never straddle a missing day, and only rows with
label_is_valid are used as window ends / targets (filtering those rows out
instead would punch a 1-hour hole into every day). Unlike the neuralforecast members, this is a plain
supervised multi-task model (direction classification + h1 return
regression) trained with a standard PyTorch loop, not a forecasting model.

Usage:
    python -m models.ssm_challenger.train --val-days 7 --epochs 10 --upload
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from common.cv import purged_walk_forward_splits
from common.data import load_days
from common.feature_contract import CLASSIFICATION_TARGET, FEATURE_COLUMNS, VALIDITY_COLUMN
from common.windowing import (
    LazyWindowDataset,
    apply_feature_scaler,
    feature_matrix,
    fit_feature_scaler,
    window_end_indices,
)
from models.ssm_challenger.model import HAVE_MAMBA_SSM, SSMChallenger

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("models.ssm_challenger.train")

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
MEMBER_NAME = "ssm-challenger"
REGRESSION_TARGET = "label_return_h1"
EARLY_STOP_FRACTION = 0.1


def build_datasets(df, train_end: int, val_start: int, val_end: int, lookback: int, stride: int):
    """(train_ds, early_stop_ds, val_ds, ts_val, scaler) over the unfiltered
    frame `df`. Train windows end in [0, es_start - PURGE_BARS), early-stop
    windows in the training window's last EARLY_STOP_FRACTION [es_start,
    train_end) — that slice alone picks the best epoch, so the val window
    [val_start, val_end) is scored exactly once. A val window's lookback may
    reach back into the purge/embargo zone (features only, exactly what live
    inference would see); its label never does. Features are standardized
    with statistics from rows [0, train_end) only, and every dataset shares
    that one matrix. The h1 return target is divided by its training std so
    the regression loss isn't ~1e-6 next to cross-entropy's ~1."""
    raw = feature_matrix(df)
    scaler = fit_feature_scaler(raw, train_end)
    features = apply_feature_scaler(raw, scaler)
    del raw
    y_cls = (df[CLASSIFICATION_TARGET].fill_null(0).cast(pl.Int64).to_numpy() + 1).astype(np.int64)
    y_reg = df[REGRESSION_TARGET].fill_null(0.0).fill_nan(0.0).cast(pl.Float32).to_numpy().astype(np.float32)
    return_scale = float(np.std(y_reg[:train_end])) or 1.0
    y_reg = (y_reg / return_scale).astype(np.float32)

    common = dict(lookback=lookback, interval_seconds=config.PRIMARY_INTERVAL_SECONDS, stride=stride,
                  valid_col=VALIDITY_COLUMN, nonnull_cols=(CLASSIFICATION_TARGET, REGRESSION_TARGET))
    es_start = train_end - int(train_end * EARLY_STOP_FRACTION)
    fit_ends = window_end_indices(df, start_row=0, end_row=es_start - config.PURGE_BARS, **common)
    es_ends = window_end_indices(df, start_row=es_start, end_row=train_end, **common)
    val_ends = window_end_indices(df, start_row=val_start, end_row=val_end, **common)
    if len(fit_ends) == 0 or len(es_ends) == 0 or len(val_ends) == 0:
        raise ValueError(f"ssm-challenger: no usable windows (train={len(fit_ends)}, early_stop={len(es_ends)}, "
                         f"val={len(val_ends)}) — too little contiguous data for lookback={lookback}")
    ts_val = df["ts"].to_numpy()[val_ends]
    scaler["return_scale"] = return_scale
    return (LazyWindowDataset(features, fit_ends, lookback, y_cls, y_reg),
            LazyWindowDataset(features, es_ends, lookback, y_cls, y_reg),
            LazyWindowDataset(features, val_ends, lookback, y_cls, y_reg), ts_val, scaler)


def _unwrap(model: nn.Module) -> nn.Module:
    """Returns the underlying module when wrapped in nn.DataParallel, else
    the model itself — state_dicts must always be saved/compared in this
    unwrapped form (DataParallel prefixes every key with "module."), so a
    checkpoint written from a 1-GPU/CPU run can still be resumed on a
    2-GPU run and vice versa."""
    return model.module if isinstance(model, nn.DataParallel) else model


def _save_inprogress(path: Path, epoch: int, model: nn.Module, optimizer, best_val_loss: float,
                      best_state: dict | None, fingerprint: dict) -> None:
    torch.save({
        "epoch": epoch,
        "model_state_dict": _unwrap(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "best_state": best_state,
        "fingerprint": fingerprint,
    }, path)


def train_and_eval(
    train_ds: LazyWindowDataset, es_ds: LazyWindowDataset, val_ds: LazyWindowDataset, lookback: int, stride: int, epochs: int, batch_size: int,
    lr: float, device: str, gpus: int | None = None, resume: bool = True,
    resume_path: Path | None = None, save_every: int = 1,
    commit_to_hf: bool = False, member_name: str | None = None,
) -> tuple[SSMChallenger, dict, np.ndarray]:
    n_features = int(train_ds.features.shape[-1])
    log.info("Windows: train=%d early_stop=%d val=%d (mamba_ssm available=%s)",
             len(train_ds), len(es_ds), len(val_ds), HAVE_MAMBA_SSM)

    # scaled_inputs/early_stop_slice: a pre-fix inprogress.pt (raw inputs,
    # epochs picked on val) must not be resumed into this run.
    fingerprint = {"n_features": n_features, "lookback": lookback, "stride": stride,
                    "lr": lr, "batch_size": batch_size, "scaled_inputs": True,
                    "early_stop_slice": EARLY_STOP_FRACTION}

    model = SSMChallenger(n_features=n_features).to(device)
    n_gpus = torch.cuda.device_count() if gpus is None else gpus
    if device == "cuda" and n_gpus > 1:
        log.info("Wrapping model in nn.DataParallel across %d GPUs", n_gpus)
        model = nn.DataParallel(model, device_ids=list(range(n_gpus)))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    cls_loss_fn = nn.CrossEntropyLoss()
    reg_loss_fn = nn.MSELoss()

    pin = device == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin)
    es_loader = DataLoader(es_ds, batch_size=batch_size, shuffle=False, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=pin)

    start_epoch = 0
    best_val_loss = float("inf")
    best_state = None
    if resume and resume_path is not None and resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device)
        if ckpt.get("fingerprint") != fingerprint:
            log.warning("Ignoring %s: hyperparameters changed (%s vs current %s) — starting fresh",
                        resume_path, ckpt.get("fingerprint"), fingerprint)
        else:
            # A run that already finished every epoch (session died before
            # the final save) resumes with an empty loop straight to its
            # best weights instead of retraining from scratch.
            _unwrap(model).load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_val_loss = ckpt["best_val_loss"]
            best_state = ckpt["best_state"]
            log.info("Resumed from %s at epoch %d (best_val_loss so far=%.4f)",
                      resume_path, start_epoch, best_val_loss)

    for epoch in range(start_epoch, epochs):
        model.train()
        for xb, yb_cls, yb_reg in train_loader:
            xb, yb_cls, yb_reg = xb.to(device), yb_cls.to(device), yb_reg.to(device)
            optimizer.zero_grad()
            logits, reg_pred = model(xb)
            loss = cls_loss_fn(logits, yb_cls) + reg_loss_fn(reg_pred, yb_reg)
            loss.backward()
            optimizer.step()

        model.eval()
        es_losses = []
        with torch.no_grad():
            for xb, yb_cls, yb_reg in es_loader:  # epoch selection never sees the val window
                xb, yb_cls, yb_reg = xb.to(device), yb_cls.to(device), yb_reg.to(device)
                logits, reg_pred = model(xb)
                loss = cls_loss_fn(logits, yb_cls) + reg_loss_fn(reg_pred, yb_reg)
                es_losses.append(loss.item())
        val_loss = float(np.mean(es_losses)) if es_losses else float("inf")
        log.info("Epoch %d: early_stop_loss=%.4f", epoch, val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in _unwrap(model).state_dict().items()}

        if resume_path is not None and (epoch % save_every == 0 or epoch == epochs - 1):
            resume_path.parent.mkdir(parents=True, exist_ok=True)
            _save_inprogress(resume_path, epoch, model, optimizer, best_val_loss, best_state, fingerprint)
            if commit_to_hf and member_name is not None:
                # Pushes the in-progress checkpoint to the private HF model
                # repo (see common/hf_model_upload.py) so a Kaggle session
                # that gets interrupted mid-training (session time limits,
                # disconnects) can be resumed from a *fresh* VM — Kaggle's
                # local disk does not carry over between separate sessions,
                # but this HF repo does.
                from common.hf_model_upload import upload_best_checkpoint

                try:
                    upload_best_checkpoint(member_name, [resume_path])
                    log.info("Committed in-progress checkpoint (epoch %d) to HF", epoch)
                except Exception:
                    log.exception("Failed to commit in-progress checkpoint to HF at epoch %d — continuing locally", epoch)

    if best_state is not None:
        _unwrap(model).load_state_dict(best_state)
    model = _unwrap(model)

    # Recompute predictions with the best (not necessarily last-epoch)
    # weights so the reported metrics match the checkpoint actually saved.
    model.eval()
    final_pred_cls, final_true_cls, final_proba = [], [], []
    with torch.no_grad():
        for xb, yb_cls, _ in val_loader:
            xb, yb_cls = xb.to(device), yb_cls.to(device)
            logits, _ = model(xb)
            final_pred_cls.append(logits.argmax(dim=-1).cpu().numpy())
            final_true_cls.append(yb_cls.cpu().numpy())
            final_proba.append(torch.softmax(logits, dim=-1).cpu().numpy())

    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    pred_cls = np.concatenate(final_pred_cls) if final_pred_cls else np.array([])
    true_cls = np.concatenate(final_true_cls) if final_true_cls else np.array([])
    val_proba = np.concatenate(final_proba) if final_proba else np.zeros((0, 3), dtype=np.float32)
    metrics = {
        "best_early_stop_loss": best_val_loss,
        "accuracy": float(accuracy_score(true_cls, pred_cls)) if len(true_cls) else None,
        "balanced_accuracy": float(balanced_accuracy_score(true_cls, pred_cls)) if len(true_cls) else None,
        "f1_macro": float(f1_score(true_cls, pred_cls, average="macro", zero_division=0)) if len(true_cls) else None,
        "n_train_windows": len(train_ds),
        "n_early_stop_windows": len(es_ds),
        "n_val_windows": len(val_ds),
        "mamba_ssm_available": HAVE_MAMBA_SSM,
    }
    return model, metrics, val_proba


def run(val_days: int, max_days: int | None, lookback: int, stride: int, epochs: int,
        batch_size: int, lr: float, upload: bool, gpus: int | None = None,
        resume: bool = True, save_every: int = 1, commit_to_hf: bool = False) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = load_days(max_days=max_days)  # NOT filter_valid — see module docstring
    val_size = val_days * config.BARS_PER_DAY
    split = purged_walk_forward_splits(df, n_folds=1, val_size=val_size)[0]
    fold = split.fold
    train_ds, es_ds, val_ds, ts_val, scaler = build_datasets(df, fold.train_end, fold.val_start, fold.val_end,
                                                            lookback, stride)

    resume_path = CHECKPOINT_DIR / "inprogress.pt"
    if resume and commit_to_hf and not resume_path.exists():
        from common.hf_model_upload import download_checkpoint_file

        pulled = download_checkpoint_file(MEMBER_NAME, "inprogress.pt", CHECKPOINT_DIR)
        if pulled is not None:
            log.info("Pulled in-progress checkpoint from HF to resume: %s", pulled)

    model, metrics, val_proba = train_and_eval(
        train_ds, es_ds, val_ds, lookback, stride, epochs, batch_size, lr, device,
        gpus=gpus, resume=resume, resume_path=resume_path, save_every=save_every,
        commit_to_hf=commit_to_hf, member_name=MEMBER_NAME,
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = CHECKPOINT_DIR / "ssm_challenger.pt"
    torch.save({"state_dict": model.state_dict(), "n_features": len(model.input_proj.weight[0]),
                "feature_columns": FEATURE_COLUMNS, "feature_cols": FEATURE_COLUMNS, "lookback": lookback, "feature_mean": scaler["mean"],
                "feature_scale": scaler["scale"], "return_scale": scaler["return_scale"]}, model_path)

    training_data_path = CHECKPOINT_DIR / "training_data_snapshot.parquet"
    split.train.write_parquet(training_data_path, compression="snappy")
    (CHECKPOINT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log.info("Saved checkpoint to %s", model_path)

    # Training reached its final epoch cleanly, so the in-progress marker no
    # longer represents unfinished work — remove it so a later --resume run
    # doesn't try to resume a completed run (see train_and_eval's fingerprint/
    # epoch guard, which would just ignore it anyway, but this keeps local +
    # HF state unambiguous).
    resume_path.unlink(missing_ok=True)

    from common.oof import save_oof

    save_oof(MEMBER_NAME, ts_val, val_proba)

    if upload:
        from common.hf_model_upload import upload_best_checkpoint, upload_training_data_snapshot

        repo_id = upload_best_checkpoint(MEMBER_NAME, [model_path, CHECKPOINT_DIR / "metrics.json"], metrics=metrics)
        data_repo_id = upload_training_data_snapshot(MEMBER_NAME, [training_data_path])
        log.info("Uploaded checkpoint to %s and training data to %s", repo_id, data_repo_id)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-days", type=int, default=config.VAL_DAYS)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--lookback", type=int, default=config.LOOKBACK_PERIODS)
    parser.add_argument("--stride", type=int, default=10, help="window-start subsampling stride")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--gpus", type=int, default=None,
                         help="GPU count for training (nn.DataParallel across this many); default auto-detects via torch.cuda.device_count()")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True,
                         help="resume from checkpoints/inprogress.pt if present and hyperparameters match (default: on)")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="ignore any existing in-progress checkpoint and start fresh")
    parser.add_argument("--save-every", type=int, default=1, help="save (and, with --commit-to-hf, push) the in-progress checkpoint every N epochs")
    parser.add_argument("--commit-to-hf", action="store_true",
                         help="also push the in-progress checkpoint to this member's private HF repo every --save-every epochs, "
                              "and pull it back down at startup if --resume and no local copy exists — "
                              "the mechanism for resuming across separate Kaggle sessions, whose local disk doesn't persist. "
                              "Requires HF_TOKEN (see common/hf_model_upload.py).")
    args = parser.parse_args()
    run(args.val_days, args.max_days, args.lookback, args.stride, args.epochs, args.batch_size, args.lr, args.upload,
        args.gpus, args.resume, args.save_every, args.commit_to_hf)


if __name__ == "__main__":
    main()
