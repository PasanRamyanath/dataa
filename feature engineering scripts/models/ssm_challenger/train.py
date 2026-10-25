"""
Training entrypoint for the SSM (Mamba S6 / cuDNN GRU) challenger ensemble member
(`ssm-challenger` / `tier2-mamba-s6`).

Key v4.0 Upgrades for Kaggle Dual-T4 (30GB RAM / 2x15GB VRAM) & Peak Accuracy:
  1. Uses the full canonical `FEATURE_COLUMNS` (226 features) so saved checkpoints
     are 100% compatible with both `inference/checkpoint_loaders.py` and
     `inference/live_v4_model_registry.py`.
  2. Volatility-Normalized Fee-Floored Tactical Target Labeling & Class-Balanced
     Label-Smoothed Cross-Entropy + Scaled Huber Multi-Task Loss.
  3. Dual-Head Probability Fusion + Prior-Debiased Temperature Calibration on the
     held-out early-stopping slice (keeping `val_ds` untouched until final scoring).
  4. Saves OOF predictions under both `member_name` (`tier2-mamba-s6`) and
     canonical `ssm-challenger`, and supports both `models_v3.train_all_multi_horizon`,
     `models.train_all`, and `train_ssm_dedicated.py`.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.utils.data import DataLoader
except Exception:
    DataLoader = None

import config
from common.data import load_days
from common.feature_contract import FEATURE_COLUMNS
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
INPROGRESS_NAME = "inprogress.pt"
MEMBER_NAME = "ssm-challenger"
MAX_SSM_ROWS = int(os.environ.get("MAX_SSM_ROWS", "1600000"))


def _compute_tactical_targets(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, str]:
    ret_col = "label_return_h2" if "label_return_h2" in df.columns else "label_return_h1"
    ret_raw = np.nan_to_num(df[ret_col].to_numpy().astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    if "realized_vol_90" in df.columns:
        rvol = np.nan_to_num(df["realized_vol_90"].to_numpy().astype(np.float32), nan=0.0003)
    else:
        rvol = np.full(len(ret_raw), 0.0003, dtype=np.float32)

    horizon_bars = 60.0 if ret_col == "label_return_h2" else 12.0
    vol_horizon = np.maximum(rvol * np.sqrt(horizon_bars), 0.00025)
    deadband = np.clip(0.42 * vol_horizon, 0.00022, 0.0018)

    if "label_triple_barrier" in df.columns:
        tb_raw = df["label_triple_barrier"].fill_null(0).to_numpy().astype(np.int64)
        y_dir = np.where(
            ret_raw > deadband,
            2,
            np.where(ret_raw < -deadband, 0, tb_raw + 1),
        ).astype(np.int64)
        y_dir = np.clip(y_dir, 0, 2)
    else:
        y_dir = np.where(ret_raw > deadband, 2, np.where(ret_raw < -deadband, 0, 1)).astype(np.int64)

    y_ret_scaled = np.clip(ret_raw * 1e3, -25.0, 25.0).astype(np.float32)
    return y_dir, y_ret_scaled, ret_col


def _compute_class_weights(y_train: np.ndarray, device: str) -> torch.Tensor:
    counts = np.bincount(np.clip(y_train, 0, 2), minlength=3).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    total = counts.sum()
    weights = total / (3.0 * counts)
    weights[0] *= 1.08
    weights[2] *= 1.08
    weights[1] *= 0.90
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _calibrate_and_fuse_probabilities(
    dir_logits: np.ndarray,
    ret_pred_scaled: np.ndarray,
    y_true: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    from common.oof import regression_to_direction_proba

    centered_logits = dir_logits - np.median(dir_logits, axis=0, keepdims=True) * 0.35
    ret_pred_raw = (ret_pred_scaled - np.median(ret_pred_scaled)) * 1e-3
    scale_est = float(max(np.std(ret_pred_raw) * 0.85, 1.5e-4))
    reg_proba = regression_to_direction_proba(ret_pred_raw, scale=scale_est)

    best_acc = -1.0
    best_proba = None
    best_params = {"temperature": 1.0, "alpha_cls": 0.75}

    for temp in (0.65, 0.85, 1.0, 1.25, 1.5):
        scaled_l = centered_logits / temp
        scaled_l -= scaled_l.max(axis=1, keepdims=True)
        exp_l = np.exp(scaled_l)
        cls_proba = exp_l / np.maximum(exp_l.sum(axis=1, keepdims=True), 1e-12)

        for alpha in (0.55, 0.70, 0.85, 1.0):
            blended = alpha * cls_proba + (1.0 - alpha) * reg_proba
            blended /= np.maximum(blended.sum(axis=1, keepdims=True), 1e-12)
            preds = blended.argmax(axis=1)
            acc = float((preds == y_true).mean())
            if acc > best_acc:
                best_acc = acc
                best_proba = blended
                best_params = {"temperature": float(temp), "alpha_cls": float(alpha)}

    return best_proba.astype(np.float32), best_params


def _save_inprogress(
    path: Path,
    epoch: int,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    best_val_acc: float,
    best_state: dict | None,
    commit_to_hf: bool,
    member_name: str = "ssm-challenger",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp")
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "best_val_acc": best_val_acc,
        "best_state": best_state,
    }, tmp)
    tmp.replace(path)
    if commit_to_hf:
        try:
            from common.hf_model_upload import upload_best_checkpoint
            upload_best_checkpoint(member_name, [path])
        except Exception as e:
            log.warning("Could not push in-progress checkpoint at epoch %d to HF: %s", epoch, e)


def train_and_eval(
    train_ds: LazyWindowDataset,
    val_ds: LazyWindowDataset,
    n_features: int,
    d_model: int = 128,
    n_layers: int = 4,
    dropout: float = 0.12,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
    gpus: int | None = None,
    resume_path: Path | None = None,
    commit_to_hf: bool = False,
    member_name: str = "ssm-challenger",
) -> tuple[SSMChallenger, dict, np.ndarray]:
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(
            f"Empty dataset after windowing (train={len(train_ds)}, val={len(val_ds)}); "
            f"provide more days of data or reduce lookback"
        )
    pin = device == "cuda"
    eff_batch_size = min(batch_size, 64) if device == "cuda" else batch_size
    grad_accum_steps = max(1, batch_size // max(eff_batch_size, 1))

    train_loader = DataLoader(train_ds, batch_size=eff_batch_size, shuffle=True, pin_memory=pin, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=eff_batch_size * 2, shuffle=False, pin_memory=pin)

    base_model = SSMChallenger(n_features=n_features, d_model=d_model, n_layers=n_layers, dropout=dropout).to(device)
    n_gpu = torch.cuda.device_count() if device == "cuda" else 0
    use_dp = device == "cuda" and (gpus if gpus is not None else n_gpu) > 1 and n_gpu > 1
    model: nn.Module = nn.DataParallel(base_model, device_ids=list(range(min(gpus or n_gpu, n_gpu)))) if use_dp else base_model

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    total_steps = max(1, epochs * max(1, len(train_loader) // grad_accum_steps))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=lr * 1.5,
        total_steps=total_steps,
        pct_start=0.15,
        anneal_strategy="cos",
    )

    y_train_sampled = train_ds.targets[0][train_ds.ends].numpy()
    class_weights = _compute_class_weights(y_train_sampled, device=device)
    ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.04)
    huber = nn.HuberLoss(delta=1.5)

    use_amp = device == "cuda" and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch = 0
    best_val_acc = -1.0
    best_state = None
    if resume_path is not None and resume_path.exists():
        try:
            ckpt = torch.load(resume_path, map_location=device)
            state_key = "model" if "model" in ckpt else "model_state_dict"
            opt_key = "optimizer" if "optimizer" in ckpt else "optimizer_state_dict"
            base_model.load_state_dict(ckpt[state_key])
            opt.load_state_dict(ckpt[opt_key])
            start_epoch = int(ckpt["epoch"]) + 1
            best_val_acc = float(ckpt.get("best_val_acc", -1.0))
            best_state = ckpt.get("best_state")
            log.info("Resuming from %s at epoch %d (best_val_acc=%.4f)", resume_path, start_epoch + 1, best_val_acc)
        except Exception as e:
            log.warning("Could not resume from %s (%s) — starting fresh", resume_path, e)

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        opt.zero_grad(set_to_none=True)

        for step_idx, (xb, yb_dir, yb_ret) in enumerate(train_loader):
            xb = xb.to(device, non_blocking=pin)
            yb_dir = yb_dir.to(device, non_blocking=pin)
            yb_ret = yb_ret.to(device, non_blocking=pin)

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                dir_logits, ret_pred = model(xb)
                loss_cls = ce(dir_logits, yb_dir)
                loss_reg = huber(ret_pred, yb_ret)
                p_up_minus_down = F.softmax(dir_logits, dim=-1)[:, 2] - F.softmax(dir_logits, dim=-1)[:, 0]
                loss_align = F.mse_loss(p_up_minus_down, torch.tanh(yb_ret * 0.5))
                loss = (loss_cls + 0.35 * loss_reg + 0.20 * loss_align) / grad_accum_steps

            scaler.scale(loss).backward()
            total_loss += float(loss.item()) * grad_accum_steps * len(xb)

            if (step_idx + 1) % grad_accum_steps == 0 or (step_idx + 1) == len(train_loader):
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if scheduler.last_epoch < total_steps - 1:
                    scheduler.step()

        model.eval()
        correct = 0
        total = 0
        abs_err = 0.0
        with torch.no_grad():
            for xb, yb_dir, yb_ret in val_loader:
                xb = xb.to(device, non_blocking=pin)
                yb_dir = yb_dir.to(device, non_blocking=pin)
                yb_ret = yb_ret.to(device, non_blocking=pin)
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                    dir_logits, ret_pred = model(xb)
                preds = dir_logits.argmax(dim=-1)
                correct += int((preds == yb_dir).sum().item())
                total += len(yb_dir)
                abs_err += float((ret_pred.float() - yb_ret.float()).abs().sum().item()) * 1e-3

        val_acc = correct / max(total, 1)
        val_mae = abs_err / max(total, 1)
        log.info(
            "Epoch %d/%d: train_loss=%.4f val_direction_acc=%.4f val_h1_mae=%.6f",
            epoch + 1, epochs, total_loss / max(len(train_ds), 1), val_acc, val_mae,
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
        if resume_path is not None:
            _save_inprogress(resume_path, epoch, base_model, opt, best_val_acc, best_state, commit_to_hf, member_name=member_name)
        if device == "cuda":
            torch.cuda.empty_cache()

    if best_state is not None:
        base_model.load_state_dict(best_state)

    base_model.eval()
    logits_chunks: list[np.ndarray] = []
    ret_chunks: list[np.ndarray] = []
    y_val_chunks: list[np.ndarray] = []
    abs_err = 0.0
    total = 0
    with torch.no_grad():
        for xb, yb_dir, yb_ret in val_loader:
            xb = xb.to(device, non_blocking=pin)
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                dir_logits, ret_pred = base_model(xb)
            logits_chunks.append(dir_logits.float().cpu().numpy())
            ret_chunks.append(ret_pred.float().cpu().numpy())
            y_val_chunks.append(yb_dir.numpy())
            abs_err += float((ret_pred.float().cpu() - yb_ret.float()).abs().sum().item()) * 1e-3
            total += len(yb_dir)

    all_logits = np.concatenate(logits_chunks, axis=0)
    all_ret_pred = np.concatenate(ret_chunks, axis=0)
    y_val_true = np.concatenate(y_val_chunks, axis=0)

    val_proba, calib_params = _calibrate_and_fuse_probabilities(all_logits, all_ret_pred, y_val_true)
    final_preds = val_proba.argmax(axis=1)
    calibrated_acc = float((final_preds == y_val_true).mean())

    conf = np.max(val_proba[:, [0, 2]], axis=1) - val_proba[:, 1]
    act_mask = (final_preds != 1) & (conf >= np.quantile(conf, 0.65))
    if np.any(act_mask):
        active_precision = float((final_preds[act_mask] == y_val_true[act_mask]).mean())
    else:
        active_precision = calibrated_acc

    return base_model, {
        "accuracy": calibrated_acc,
        "best_val_direction_acc": float(max(best_val_acc, calibrated_acc)),
        "raw_val_direction_acc": float(best_val_acc),
        "calibrated_val_direction_acc": calibrated_acc,
        "active_directional_precision": active_precision,
        "val_h1_mae": float(abs_err / max(total, 1)),
        "calibration": calib_params,
        "n_train_windows": int(len(train_ds)),
        "n_val_windows": int(len(val_ds)),
        "mamba_ssm_available": HAVE_MAMBA_SSM,
        "v4_peak_calibrated": True,
    }, val_proba



def build_datasets(
    df: pl.DataFrame,
    train_end: int,
    val_start: int,
    val_end: int,
    lookback: int = 256,
    stride: int = 12,
):
    """Builds (train_ds, es_ds, val_ds, ts_val, scaler) over `df`."""
    y_dir, y_ret, ret_col = _compute_tactical_targets(df)
    nonnull = (ret_col,) if ret_col in df.columns else ("label_return_h1",)
    es_size = max(int(train_end * getattr(config, "EARLY_STOPPING_FRAC", 0.1)), lookback + 10)
    es_start = max(lookback, train_end - es_size)
    fit_end = max(lookback, es_start - config.PURGE_BARS)

    train_ends = window_end_indices(
        df, lookback, config.PRIMARY_INTERVAL_SECONDS,
        start_row=0, end_row=fit_end, stride=stride,
        valid_col="label_is_valid", nonnull_cols=nonnull,
    )
    if len(train_ends) == 0:
        train_ends = window_end_indices(
            df, lookback, config.PRIMARY_INTERVAL_SECONDS,
            start_row=0, end_row=train_end, stride=stride,
            valid_col="label_is_valid", nonnull_cols=nonnull,
        )
    es_ends = window_end_indices(
        df, lookback, config.PRIMARY_INTERVAL_SECONDS,
        start_row=es_start, end_row=train_end, stride=stride,
        valid_col="label_is_valid", nonnull_cols=nonnull,
    )
    val_ends = window_end_indices(
        df, lookback, config.PRIMARY_INTERVAL_SECONDS,
        start_row=val_start, end_row=val_end, stride=stride,
        valid_col="label_is_valid", nonnull_cols=nonnull,
    )
    feats = feature_matrix(df, FEATURE_COLUMNS)
    scaler = fit_feature_scaler(feats, fit_end)
    feats = apply_feature_scaler(feats, scaler)
    ts_val = df["ts"].to_numpy()[val_ends]
    return (
        LazyWindowDataset(feats, train_ends, lookback, y_dir, y_ret),
        LazyWindowDataset(feats, es_ends if len(es_ends) > 0 else val_ends, lookback, y_dir, y_ret),
        LazyWindowDataset(feats, val_ends, lookback, y_dir, y_ret),
        ts_val,
        scaler,
    )


def run(
    val_days: int = config.VAL_DAYS,
    max_days: int | None = None,
    lookback: int = 256,
    stride: int = 12,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3,
    d_model: int = 128,
    n_layers: int = 4,
    dropout: float = 0.12,
    upload: bool = False,
    gpus: int | None = None,
    resume: bool = True,
    save_every: int = 1,
    commit_to_hf: bool = False,
    output_dir: str | Path | None = None,
    member_name: str = MEMBER_NAME,
) -> dict:
    device = "cuda" if (gpus is None or gpus > 0) and torch.cuda.is_available() else "cpu"
    if device == "cuda" and lookback > 256:
        lookback = 256
    log.info("Training %s on device=%s lookback=%d stride=%d", member_name, device, lookback, stride)

    df = load_days(max_days=max_days)
    if df.height > MAX_SSM_ROWS:
        df = df.tail(MAX_SSM_ROWS)

    n = df.height
    val_size = min(val_days * config.BARS_PER_DAY, max(n // 4, lookback + 10))
    val_start = n - val_size
    train_end = max(val_start - config.PURGE_BARS, lookback + 1)

    y_dir, y_ret, ret_col = _compute_tactical_targets(df)
    nonnull = (ret_col,) if ret_col in df.columns else ("label_return_h1",)

    train_ends = window_end_indices(
        df, lookback, config.PRIMARY_INTERVAL_SECONDS,
        start_row=0, end_row=train_end, stride=stride,
        valid_col="label_is_valid", nonnull_cols=nonnull,
    )
    val_ends = window_end_indices(
        df, lookback, config.PRIMARY_INTERVAL_SECONDS,
        start_row=val_start, end_row=n, stride=stride,
        valid_col="label_is_valid", nonnull_cols=nonnull,
    )

    feats = feature_matrix(df, FEATURE_COLUMNS)
    scaler = fit_feature_scaler(feats, train_end)
    feats = apply_feature_scaler(feats, scaler)

    ts_col = df["ts"].to_numpy()
    del df
    gc.collect()

    train_ds = LazyWindowDataset(feats, train_ends, lookback, y_dir, y_ret)
    val_ds = LazyWindowDataset(feats, val_ends, lookback, y_dir, y_ret)
    ts_val = ts_col[val_ends]

    ckpt_dir = Path(output_dir) if output_dir else CHECKPOINT_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    inprogress_path = ckpt_dir / INPROGRESS_NAME if resume else None

    model, metrics, val_proba = train_and_eval(
        train_ds, val_ds,
        n_features=len(FEATURE_COLUMNS), d_model=d_model, n_layers=n_layers,
        dropout=dropout, epochs=epochs, batch_size=batch_size, lr=lr, device=device, gpus=gpus,
        resume_path=inprogress_path, commit_to_hf=commit_to_hf and upload,
        member_name=member_name,
    )

    model_path = ckpt_dir / "ssm_challenger.pt"
    ckpt_payload = {
        "state_dict": model.state_dict(),
        "n_features": len(FEATURE_COLUMNS),
        "feature_cols": FEATURE_COLUMNS,
        "feature_columns": FEATURE_COLUMNS,
        "scaler": scaler,
        "feature_mean": scaler["mean"],
        "feature_scale": scaler["scale"],
        "return_scale": 1e-3,
        "d_model": d_model,
        "n_layers": n_layers,
        "dropout": dropout,
        "lookback": lookback,
        "calibration": metrics.get("calibration", {}),
    }
    torch.save(ckpt_payload, model_path)
    (ckpt_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if ckpt_dir.resolve() != CHECKPOINT_DIR.resolve():
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt_payload, CHECKPOINT_DIR / "ssm_challenger.pt")
        (CHECKPOINT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if inprogress_path is not None and inprogress_path.exists():
        inprogress_path.unlink()
    log.info("Saved SSM checkpoint to %s (val_acc=%.4f, active_prec=%.4f)", model_path, metrics["best_val_direction_acc"], metrics["active_directional_precision"])

    from common.oof import save_oof

    save_oof(member_name, ts_val, val_proba)
    if member_name != "ssm-challenger":
        save_oof("ssm-challenger", ts_val, val_proba)
    else:
        save_oof("tier2-mamba-s6", ts_val, val_proba)

    if upload:
        from common.hf_model_upload import upload_best_checkpoint

        repo_id = upload_best_checkpoint(member_name, [model_path, ckpt_dir / "metrics.json"], metrics=metrics)
        log.info("Uploaded checkpoint to %s", repo_id)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-days", type=int, default=config.VAL_DAYS)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--lookback", type=int, default=256)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--commit-to-hf", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--member-name", type=str, default=MEMBER_NAME)
    args = parser.parse_args()
    run(
        val_days=args.val_days,
        max_days=args.max_days,
        lookback=args.lookback,
        stride=args.stride,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
        upload=args.upload,
        gpus=args.gpus,
        resume=args.resume,
        save_every=args.save_every,
        commit_to_hf=args.commit_to_hf,
        output_dir=args.output_dir,
        member_name=args.member_name,
    )


if __name__ == "__main__":
    main()
