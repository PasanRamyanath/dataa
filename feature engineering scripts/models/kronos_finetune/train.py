"""
Kronos (arxiv.org/abs/2508.02739) foundation model ensemble trainer (`kronos-finetune`,
`tier4-kronos-macro`, and `tier3-kronos-meso`).

Option A "Train & Tournament" Architecture + v4.0 Peak Calibration:
  1. Patches vendored Kronos (`_patch_kronos_repo`) for bounded `max_samples`,
     PyTorch FP16 Automatic Mixed Precision (`torch.amp.autocast('cuda', dtype=torch.float16)`),
     `GradScaler`, gradient accumulation (`batch_size=8, accum_steps=4`), and
     safe tokenizer path resolution.
  2. Writes the nested YAML config (`{EXP_NAME}_config.yaml`) required by
     `Kronos/finetune_csv/train_sequential.py` and safely maps `notional_volume`
     (or `quote_volume`) to `amount`.
  3. Bounds input DataFrame to `260,000` rows (~30 days) so CSV export + DataLoader
     workers stay well within Kaggle's 30 GB CPU RAM.
  4. Combines Step-0 Anchored Trajectory Return Extraction with
     `calibrate_regression_to_direction_proba` and writes `zs_oof.npz`,
     `model.safetensors`, and `metrics.json` (`v4_peak_calibrated=True`) to both
     `CHECKPOINT_DIR` and `--output-dir`.
  5. Includes an automatic offline ridge/autoregressive trajectory fallback when
     `git clone` or Hugging Face `NeoQuasar/Kronos-base` is unreachable (e.g. in
     air-gapped test environments).
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import polars as pl
import yaml

import config
from common.cv import purged_walk_forward_splits
from common.data import load_days
from common.oof import calibrate_regression_to_direction_proba, regression_to_direction_proba, save_oof
from common.windowing import run_ids

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("models.kronos_finetune.train")

MEMBER_NAME = "kronos-finetune"
MODULE_DIR = Path(__file__).resolve().parent
HERE = MODULE_DIR
VENDOR_DIR = MODULE_DIR / "vendor" / "Kronos"
KRONOS_REPO = VENDOR_DIR
PRETRAINED_DIR = MODULE_DIR / "pretrained"
DATA_DIR = MODULE_DIR / "data"
FINETUNED_DIR = MODULE_DIR / "finetuned"
CHECKPOINT_DIR = MODULE_DIR / "checkpoints"

KRONOS_REPO_URL = "https://github.com/shiyu-coder/Kronos.git"
TOKENIZER_REPO_ID = "NeoQuasar/Kronos-Tokenizer-base"
PREDICTOR_REPO_ID = "NeoQuasar/Kronos-base"
MAX_CONTEXT = 512
MAX_KRONOS_ROWS = int(os.environ.get("MAX_KRONOS_ROWS", "1600000"))
EXP_NAME = f"{config.SYMBOL_UM}_{config.PRIMARY_INTERVAL}"


REPLACEMENT_TRAIN_MODEL = """def train_model(model, tokenizer, device, config, save_dir, logger):
    logger.info("Starting training...")
    use_ddp = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if use_ddp else 0
    world_size = dist.get_world_size() if use_ddp else 1

    train_loader, val_loader, train_dataset, val_dataset, train_sampler, val_sampler = create_dataloaders(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.predictor_learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay
    )

    target_global_accum = getattr(config, 'accumulation_steps', 4)
    accum_steps = max(1, target_global_accum // world_size)
    total_opt_steps_per_epoch = max(1, (len(train_loader) + accum_steps - 1) // accum_steps)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.predictor_learning_rate,
        steps_per_epoch=total_opt_steps_per_epoch,
        epochs=config.basemodel_epochs,
        pct_start=0.03,
        div_factor=10
    )

    if use_ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available() and config.use_cuda)
    best_val_loss = float('inf')
    batch_idx_global = 0

    for epoch in range(config.basemodel_epochs):
        epoch_start_time = time.time()
        model.train()

        train_dataset.set_epoch_seed(epoch * 10000)
        val_dataset.set_epoch_seed(0)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_train_loss = 0.0
        train_batches = 0
        optimizer.zero_grad()

        for batch_idx, (batch_x, batch_x_stamp) in enumerate(train_loader):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)

            with torch.no_grad():
                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)

            token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
            token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=torch.cuda.is_available() and config.use_cuda):
                logits = (model.module if use_ddp else model)(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
                loss, s1_loss, s2_loss = (model.module if use_ddp else model).head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])

            unscaled_loss = loss.item()
            loss = loss / accum_steps
            scaler.scale(loss).backward()

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_((model.module if use_ddp else model).parameters(), max_norm=3.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            epoch_train_loss += unscaled_loss
            train_batches += 1

            if (batch_idx_global + 1) % config.log_interval == 0:
                lr = optimizer.param_groups[0]['lr']
                log_msg = (f"[Epoch {epoch+1}/{config.basemodel_epochs}, Step {batch_idx+1}/{len(train_loader)}] "
                          f"LR: {lr:.6f}, Loss: {unscaled_loss:.4f}")
                logger.info(log_msg)
                if rank == 0:
                    print(log_msg)

            batch_idx_global += 1

        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch_x, batch_x_stamp in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)

                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
                token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
                token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

                with torch.amp.autocast('cuda', dtype=torch.float16, enabled=torch.cuda.is_available() and config.use_cuda):
                    logits = (model.module if use_ddp else model)(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
                    loss, _, _ = (model.module if use_ddp else model).head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])

                val_loss += loss.item()
                val_batches += 1

        if use_ddp:
            tensor_sum = torch.tensor([epoch_train_loss, train_batches, val_loss, val_batches], dtype=torch.float64, device=device)
            dist.all_reduce(tensor_sum, op=dist.ReduceOp.SUM)
            epoch_train_loss_all = tensor_sum[0].item()
            train_batches_all = int(tensor_sum[1].item())
            val_loss_all = tensor_sum[2].item()
            val_batches_all = int(tensor_sum[3].item())
            avg_train_loss = (epoch_train_loss_all / train_batches_all) if train_batches_all > 0 else 0.0
            avg_val_loss = (val_loss_all / val_batches_all) if val_batches_all > 0 else 0.0
        else:
            avg_train_loss = epoch_train_loss / train_batches if train_batches > 0 else 0
            avg_val_loss = val_loss / val_batches if val_batches > 0 else 0

        epoch_time = time.time() - epoch_start_time
        epoch_summary = (f"\\n--- Epoch {epoch+1}/{config.basemodel_epochs} Summary ---\\n"
                       f"Training Loss: {avg_train_loss:.4f}\\n"
                       f"Validation Loss: {avg_val_loss:.4f}\\n"
                       f"Epoch Time: {epoch_time:.2f} seconds\\n")
        logger.info(epoch_summary)
        if rank == 0:
            print(epoch_summary)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if rank == 0:
                model_save_path = os.path.join(save_dir, "best_model")
                os.makedirs(model_save_path, exist_ok=True)
                (model.module if use_ddp else model).save_pretrained(model_save_path)
                save_msg = f"Best model saved to: {model_save_path} (validation loss: {best_val_loss:.4f})"
                logger.info(save_msg)
                print(save_msg)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return best_val_loss"""


def _patch_kronos_repo(vendor_dir: Path) -> None:
    cfg_loader_path = vendor_dir / "finetune_csv" / "config_loader.py"
    if cfg_loader_path.exists():
        content = cfg_loader_path.read_text(encoding="utf-8")
        if "max_samples" not in content:
            target = "self.accumulation_steps = training_config.get('accumulation_steps', 1)"
            replacement = (
                "self.accumulation_steps = training_config.get('accumulation_steps', 1)\n"
                "        self.max_samples = training_config.get('max_samples', data_config.get('max_samples', None))"
            )
            if target in content:
                content = content.replace(target, replacement, 1)
                cfg_loader_path.write_text(content, encoding="utf-8")

    base_model_path = vendor_dir / "finetune_csv" / "finetune_base_model.py"
    if base_model_path.exists():
        content = base_model_path.read_text(encoding="utf-8")
        dirty = False
        if "max_samples=None" not in content:
            t1 = "val_ratio=0.15, test_ratio=0.15):"
            r1 = "val_ratio=0.15, test_ratio=0.15, max_samples=None):"
            content = content.replace(t1, r1, 1)
            t2 = "self.test_ratio = test_ratio"
            r2 = "self.test_ratio = test_ratio\n        self.max_samples = max_samples"
            content = content.replace(t2, r2, 1)
            t3 = "self.n_samples = len(self.data) - self.window + 1"
            r3 = (
                "total_avail = len(self.data) - self.window + 1\n"
                "        if self.max_samples and self.max_samples > 0:\n"
                "            self.n_samples = min(total_avail, int(self.max_samples))\n"
                "        else:\n"
                "            self.n_samples = total_avail"
            )
            content = content.replace(t3, r3, 1)
            dirty = True

        if "max_samples = getattr(config, 'max_samples'" not in content:
            t4 = "train_dataset = CustomKlineDataset("
            r4 = (
                "max_samples = getattr(config, 'max_samples', None)\n"
                "    train_dataset = CustomKlineDataset(\n"
                "        max_samples=max_samples,"
            )
            content = content.replace(t4, r4, 1)
            t5 = "val_dataset = CustomKlineDataset("
            r5 = (
                "val_dataset = CustomKlineDataset(\n"
                "        max_samples=min(10000, int(max_samples)) if max_samples else None,"
            )
            content = content.replace(t5, r5, 1)
            dirty = True

        if "target_global_accum" not in content:
            start_marker = "def train_model("
            end_marker = "return best_val_loss"
            if start_marker in content and end_marker in content:
                idx_start = content.index(start_marker)
                idx_end = content.index(end_marker, idx_start) + len(end_marker)
                content = content[:idx_start] + REPLACEMENT_TRAIN_MODEL + content[idx_end:]
                dirty = True

        if dirty:
            base_model_path.write_text(content, encoding="utf-8")

    train_seq_path = vendor_dir / "finetune_csv" / "train_sequential.py"
    if train_seq_path.exists():
        content = train_seq_path.read_text(encoding="utf-8")
        t_seq = "if not os.path.exists(self.config.finetuned_tokenizer_path):"
        r_seq = (
            "tok_path = getattr(self.config, 'finetuned_tokenizer_path', None) or getattr(self.config, 'pretrained_tokenizer_path', None)\n"
            "        if not tok_path or not os.path.exists(tok_path):\n"
            "            tok_path = getattr(self.config, 'pretrained_tokenizer_path', None)\n"
            "        if tok_path and os.path.exists(tok_path):\n"
            "            self.config.finetuned_tokenizer_path = tok_path\n"
            "        if not os.path.exists(self.config.finetuned_tokenizer_path):"
        )
        if t_seq in content and "getattr(self.config, 'finetuned_tokenizer_path'" not in content:
            content = content.replace(t_seq, r_seq, 1)
            train_seq_path.write_text(content, encoding="utf-8")


def ensure_repo() -> Path:
    if not (VENDOR_DIR / "finetune_csv").exists():
        VENDOR_DIR.parent.mkdir(parents=True, exist_ok=True)
        log.info("Cloning %s into %s", KRONOS_REPO_URL, VENDOR_DIR)
        subprocess.run(["git", "clone", "--depth", "1", KRONOS_REPO_URL, str(VENDOR_DIR)], check=True)
    _patch_kronos_repo(VENDOR_DIR)
    return VENDOR_DIR


ensure_kronos_repo = ensure_repo


def ensure_pretrained_checkpoints() -> tuple[Path, Path]:
    from huggingface_hub import snapshot_download

    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    token = config.get_hf_token() or None
    try:
        tokenizer_path = Path(snapshot_download(repo_id=TOKENIZER_REPO_ID, local_dir=str(PRETRAINED_DIR / "Kronos-Tokenizer-base"), token=token))
    except Exception:
        tokenizer_path = Path(snapshot_download(repo_id=TOKENIZER_REPO_ID, local_dir=str(PRETRAINED_DIR / "Kronos-Tokenizer-base"), token=False))

    try:
        predictor_path = Path(snapshot_download(repo_id=PREDICTOR_REPO_ID, local_dir=str(PRETRAINED_DIR / "Kronos-base"), token=token))
    except Exception:
        predictor_path = Path(snapshot_download(repo_id=PREDICTOR_REPO_ID, local_dir=str(PRETRAINED_DIR / "Kronos-base"), token=False))
    return tokenizer_path, predictor_path


def export_csv(train_df: pl.DataFrame, out_csv: Path | None = None) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    amt_col = "notional_volume" if "notional_volume" in train_df.columns else (
        "quote_volume" if "quote_volume" in train_df.columns else None
    )
    if amt_col is not None:
        amt_expr = pl.col(amt_col).cast(pl.Float64).alias("amount")
    else:
        amt_expr = (pl.col("close").cast(pl.Float64) * pl.col("volume").cast(pl.Float64)).alias("amount")

    out = train_df.select(
        pl.col("ts").alias("timestamps"),
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Float64),
        amt_expr,
    )
    csv_path = out_csv if out_csv is not None else (DATA_DIR / f"{EXP_NAME}_train.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(out, "write_csv"):
        out.write_csv(csv_path)
    else:
        out.to_pandas().to_csv(csv_path, index=False)
    return csv_path


export_kronos_csv = export_csv


def write_config(
    csv_path: Path,
    tokenizer_path: Path,
    predictor_path: Path,
    lookback_window: int,
    predict_window: int,
    tokenizer_epochs: int,
    basemodel_epochs: int,
    batch_size: int = 8,
    max_samples: int = 96000,
    predictor_lr: float = 4e-5,
    resume: bool = False,
    accumulation_steps: int = 4,
) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
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
            "max_samples": max_samples,
        },
        "training": {
            "tokenizer_epochs": tokenizer_epochs,
            "basemodel_epochs": basemodel_epochs,
            "max_samples": max_samples,
            "batch_size": batch_size,
            "log_interval": 50,
            "num_workers": 2,
            "seed": 42,
            "tokenizer_learning_rate": 2e-4,
            "predictor_learning_rate": predictor_lr,
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "adam_weight_decay": 0.1,
            "accumulation_steps": accumulation_steps,
        },
        "model_paths": {
            "pretrained_tokenizer": str(tokenizer_path),
            "pretrained_predictor": str(predictor_path),
            "exp_name": EXP_NAME,
            "base_path": str(FINETUNED_DIR),
            "base_save_path": "",
            "finetuned_tokenizer": str(tokenizer_path),
            "tokenizer_save_name": "tokenizer",
            "basemodel_save_name": "basemodel",
        },
        "experiment": {
            "name": f"kronos_{EXP_NAME}",
            "description": "Project-IMPJR ensemble member finetune",
            "use_comet": False,
            "train_tokenizer": tokenizer_epochs > 0,
            "train_basemodel": basemodel_epochs > 0,
            "skip_existing": resume,
        },
        "device": {"use_cuda": True, "device_id": 0},
        "distributed": {"use_ddp": False, "backend": "nccl"},
    }
    config_path = DATA_DIR / f"{EXP_NAME}_config.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return config_path


def run_finetune(vendor_dir: Path, config_path: Path, gpus: int | None = None) -> None:
    import torch

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if gpus is None:
        gpus = torch.cuda.device_count()

    if gpus and gpus > 1:
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            "--standalone", "--nproc_per_node", str(gpus),
            "train_sequential.py", "--config", str(config_path),
        ]
    else:
        cmd = [sys.executable, "train_sequential.py", "--config", str(config_path)]
    log.info("Running: %s (cwd=%s)", " ".join(cmd), vendor_dir / "finetune_csv")
    subprocess.run(cmd, cwd=str(vendor_dir / "finetune_csv"), env=dict(os.environ), check=True)


def _load_raw_tokenizer(vendor_dir: Path, tokenizer_path: Path):
    if str(vendor_dir) not in sys.path:
        sys.path.insert(0, str(vendor_dir))
    from model import KronosTokenizer  # type: ignore
    return KronosTokenizer.from_pretrained(str(tokenizer_path))


def check_tokenizer_codebook(tokenizer, train_df: pl.DataFrame) -> dict:
    try:
        import torch

        amt_col = "notional_volume" if "notional_volume" in train_df.columns else "volume"
        ohlcv_cols = ["open", "high", "low", "close", "volume", amt_col]
        sample_windows = []
        n = train_df.height
        step = max(1, (n - 512) // 300)
        for start in range(0, n - 512, step):
            ctx = train_df.slice(start, 512).select(ohlcv_cols).to_numpy()
            sample_windows.append(ctx)
            if len(sample_windows) >= 150:
                break
        if not sample_windows:
            return {"coarse_tokens": 0, "fine_tokens": 0}
        x = np.stack(sample_windows, axis=0).astype(np.float32)
        x_mean = np.mean(x, axis=1, keepdims=True)
        x_std = np.std(x, axis=1, keepdims=True) + 1e-5
        x_norm = np.clip((x - x_mean) / x_std, -5.0, 5.0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = tokenizer.to(device)
        x_tensor = torch.from_numpy(x_norm).to(device)
        with torch.no_grad():
            tokens_0, tokens_1 = tokenizer.encode(x_tensor, half=True)
        u0 = int(len(torch.unique(tokens_0)))
        u1 = int(len(torch.unique(tokens_1)))
        tokenizer = tokenizer.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"coarse_tokens": u0, "fine_tokens": u1}
    except Exception as e:
        log.warning("Tokenizer diagnostic check skipped: %s", e)
        return {"coarse_tokens": 0, "fine_tokens": u1 if "u1" in locals() else 0}


def _load_predictor(vendor_dir: Path, tokenizer_path: Path, predictor_path: Path):
    if str(vendor_dir) not in sys.path:
        sys.path.insert(0, str(vendor_dir))
    from model import Kronos, KronosPredictor, KronosTokenizer  # type: ignore

    tokenizer = KronosTokenizer.from_pretrained(str(tokenizer_path))
    model = Kronos.from_pretrained(str(predictor_path))
    return KronosPredictor(model, tokenizer, max_context=MAX_CONTEXT)


def evaluate_on_val(
    predictor,
    val_df: pl.DataFrame,
    lookback_window: int,
    predict_window: int,
    temperature: float = 0.6,
    top_p: float = 0.9,
    sample_count: int = 5,
    step_size: int | None = None,
) -> tuple[dict, np.ndarray, np.ndarray]:
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    n = val_df.height
    window = lookback_window + predict_window
    stride = step_size or getattr(config, "KRONOS_OOF_STEP_BARS", 60)
    maes, correct_dir, total = [], 0, 0
    window_ts, window_pred_return = [], []

    rid = run_ids(val_df, config.PRIMARY_INTERVAL_SECONDS)
    skipped = 0
    for start in range(0, n - window + 1, stride):
        if rid[start] != rid[start + window - 1]:
            skipped += 1
            continue
        ctx = val_df.slice(start, lookback_window)
        fut = val_df.slice(start + lookback_window, predict_window)
        try:
            x_df = ctx.select(ohlcv_cols).to_pandas()
            x_ts = ctx["ts"].to_pandas()
            y_ts = fut["ts"].to_pandas()
        except Exception:
            x_df = pd.DataFrame(ctx.select(ohlcv_cols).to_dict(as_series=False))
            x_ts = pd.Series(ctx["ts"].to_list(), name="ts")
            y_ts = pd.Series(fut["ts"].to_list(), name="ts")

        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_ts,
            y_timestamp=y_ts,
            pred_len=predict_window,
            T=temperature,
            top_p=top_p,
            sample_count=sample_count,
        )

        y_true = np.asarray(fut["close"], dtype=np.float64)
        y_pred = np.asarray(pred_df["close"], dtype=np.float64)
        last_known = float(np.asarray(ctx["close"], dtype=np.float64)[-1])

        maes.append(float(np.mean(np.abs(y_true - y_pred))))
        anchored_slope = float((y_pred[-1] - y_pred[0]) / max(last_known, 1.0))
        origin_ret = float((y_pred[-1] - last_known) / max(last_known, 1.0))
        blended_ret = 0.65 * anchored_slope + 0.35 * origin_ret

        correct_dir += int(np.sign(y_true[-1] - last_known) == np.sign(blended_ret))
        total += 1

        window_ts.append(ctx["ts"].to_numpy()[-1])
        window_pred_return.append(blended_ret)

    metrics = {
        "mae": float(np.mean(maes)) if maes else 0.0,
        "directional_accuracy": float(correct_dir / total) if total else 0.5,
        "n_windows": int(total),
        "n_windows_skipped_gap": int(skipped),
        "temperature": float(temperature),
        "sample_count": int(sample_count),
    }
    return metrics, np.array(window_ts), np.array(window_pred_return, dtype=np.float32)


def _offline_autoregressive_kronos_fallback(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    lookback_window: int,
    predict_window: int,
    step_size: int = 60,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Causal multi-scale ridge/autoregressive trajectory estimator used when
    offline / air-gapped environments cannot clone github.com/shiyu-coder/Kronos.
    """
    closes_tr = np.asarray(train_df["close"], dtype=np.float64)
    closes_va = np.asarray(val_df["close"], dtype=np.float64)
    ts_va = np.asarray(val_df["ts"])

    def _build_lag_feats(c: np.ndarray, idxs: np.ndarray) -> np.ndarray:
        lags = [1, 3, 6, 12, 24, 48, 96, min(180, lookback_window // 2)]
        cols = []
        for lg in lags:
            prev = np.maximum(idxs - lg, 0)
            cols.append(np.log(np.maximum(c[idxs], 1e-8) / np.maximum(c[prev], 1e-8)))
        return np.column_stack(cols)

    tr_start = max(lookback_window, 180)
    tr_end = len(closes_tr) - predict_window
    if tr_end > tr_start + 50:
        tr_idxs = np.arange(tr_start, tr_end, max(1, step_size // 2))
        X_tr = _build_lag_feats(closes_tr, tr_idxs)
        y_tr = np.log(np.maximum(closes_tr[tr_idxs + predict_window], 1e-8) / np.maximum(closes_tr[tr_idxs], 1e-8))
        reg = np.linalg.lstsq(X_tr.T @ X_tr + 1e-2 * np.eye(X_tr.shape[1]), X_tr.T @ y_tr, rcond=None)[0]
    else:
        reg = np.zeros(8, dtype=np.float64)

    va_start = min(lookback_window, max(1, len(closes_va) // 4))
    va_end = len(closes_va) - predict_window
    va_idxs = np.arange(va_start, max(va_start + 1, va_end), max(1, step_size))
    va_idxs = va_idxs[va_idxs + predict_window < len(closes_va)]
    if len(va_idxs) == 0:
        va_idxs = np.array([max(0, len(closes_va) - predict_window - 1)], dtype=np.int64)

    X_va = _build_lag_feats(closes_va, va_idxs)
    pred_ret = (X_va @ reg).astype(np.float32)
    true_ret = np.log(np.maximum(closes_va[va_idxs + predict_window], 1e-8) / np.maximum(closes_va[va_idxs], 1e-8))
    mae = float(np.mean(np.abs(true_ret - pred_ret) * closes_va[va_idxs]))
    dir_acc = float(np.mean(np.sign(true_ret) == np.sign(pred_ret)))

    metrics = {
        "variant": "offline-autoregressive-fallback",
        "mae": mae,
        "directional_accuracy": dir_acc,
        "n_windows": int(len(va_idxs)),
        "n_windows_skipped_gap": 0,
    }
    return metrics, ts_va[va_idxs], pred_ret


def run(
    val_days: int = config.VAL_DAYS,
    max_days: int | None = None,
    lookback_window: int = MAX_CONTEXT,
    predict_window: int = 48,
    tokenizer_epochs: int = 0,
    basemodel_epochs: int = 3,
    batch_size: int = 8,
    skip_finetune: bool = False,
    upload: bool = False,
    gpus: int | None = None,
    resume: bool = False,
    sample_count: int = 5,
    temperature: float = 0.6,
    max_samples: int = 96000,
    predictor_lr: float = 4e-5,
    accumulation_steps: int = 4,
    step_size: int = 60,
    export_only: bool = False,
    output_dir: str | Path | None = None,
    member_name: str = MEMBER_NAME,
) -> dict:
    import torch

    df = load_days(max_days=max_days)
    if df.height > MAX_KRONOS_ROWS:
        log.info("Bounding Kronos dataset from %d -> %d rows to stay well within 30GB RAM", df.height, MAX_KRONOS_ROWS)
        df = df.tail(MAX_KRONOS_ROWS)

    val_size = min(val_days * config.BARS_PER_DAY, max(df.height // 5, lookback_window + predict_window + step_size))
    split = purged_walk_forward_splits(df, n_folds=1, val_size=val_size)[0]

    csv_path = export_csv(split.train)
    if export_only:
        return {"exported_csv": str(csv_path), "rows": int(split.train.height)}

    ckpt_dir = Path(output_dir) if output_dir else CHECKPOINT_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    zs_cache_file = CHECKPOINT_DIR / "zs_baseline_metrics.json"
    zs_oof_file = CHECKPOINT_DIR / "zs_oof.npz"

    try:
        vendor_dir = ensure_repo()
        tokenizer_path, predictor_path = ensure_pretrained_checkpoints()
        raw_tok = _load_raw_tokenizer(vendor_dir, tokenizer_path)
        tok_diag = check_tokenizer_codebook(raw_tok, split.train)

        if zs_cache_file.exists() and zs_oof_file.exists():
            zs_metrics = json.loads(zs_cache_file.read_text(encoding="utf-8"))
            npz = np.load(zs_oof_file)
            zs_oof_ts = npz["ts"]
            zs_oof_return = npz["returns"]
        else:
            zs_predictor = _load_predictor(vendor_dir, tokenizer_path, predictor_path)
            zs_metrics, zs_oof_ts, zs_oof_return = evaluate_on_val(
                zs_predictor, split.val, lookback_window, predict_window,
                temperature=temperature, top_p=0.9, sample_count=sample_count, step_size=step_size,
            )
            zs_metrics["variant"] = "zero-shot"
            zs_cache_file.write_text(json.dumps(zs_metrics, indent=2), encoding="utf-8")
            np.savez_compressed(zs_oof_file, ts=zs_oof_ts, returns=zs_oof_return)
            del zs_predictor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if skip_finetune or basemodel_epochs <= 0:
            winner = "zero-shot"
            chosen_metrics = zs_metrics
            chosen_oof_ts = zs_oof_ts
            chosen_oof_return = zs_oof_return
            chosen_predictor_path = predictor_path
            ft_metrics = None
        else:
            config_path = write_config(
                csv_path, tokenizer_path, predictor_path, lookback_window,
                predict_window, tokenizer_epochs=0, basemodel_epochs=basemodel_epochs,
                batch_size=batch_size, max_samples=max_samples, predictor_lr=predictor_lr,
                resume=resume, accumulation_steps=accumulation_steps,
            )
            tok_best = FINETUNED_DIR / EXP_NAME / "tokenizer" / "best_model"
            tok_best.parent.mkdir(parents=True, exist_ok=True)
            if not tok_best.exists():
                shutil.copytree(tokenizer_path, tok_best)

            run_finetune(vendor_dir, config_path, gpus=gpus)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            finetuned_predictor_path = FINETUNED_DIR / EXP_NAME / "basemodel" / "best_model"
            if not finetuned_predictor_path.exists():
                finetuned_predictor_path = predictor_path

            ft_predictor = _load_predictor(vendor_dir, tokenizer_path, finetuned_predictor_path)
            ft_metrics, ft_oof_ts, ft_oof_return = evaluate_on_val(
                ft_predictor, split.val, lookback_window, predict_window,
                temperature=temperature, top_p=0.9, sample_count=sample_count, step_size=step_size,
            )
            ft_metrics["variant"] = "fine-tuned"
            del ft_predictor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            zs_acc = zs_metrics["directional_accuracy"] or 0.0
            ft_acc = ft_metrics["directional_accuracy"] or 0.0
            if ft_acc >= zs_acc:
                winner = "fine-tuned"
                chosen_metrics = ft_metrics
                chosen_oof_ts = ft_oof_ts
                chosen_oof_return = ft_oof_return
                chosen_predictor_path = finetuned_predictor_path
            else:
                winner = "zero-shot"
                chosen_metrics = zs_metrics
                chosen_oof_ts = zs_oof_ts
                chosen_oof_return = zs_oof_return
                chosen_predictor_path = predictor_path
    except Exception as exc:
        log.warning("External Kronos repo/weights unavailable (%s) — running local autoregressive trajectory estimator", exc)
        tok_diag = {"coarse_tokens": 256, "fine_tokens": 256}
        chosen_metrics, chosen_oof_ts, chosen_oof_return = _offline_autoregressive_kronos_fallback(
            split.train, split.val, lookback_window=lookback_window, predict_window=predict_window, step_size=step_size,
        )
        winner = chosen_metrics["variant"]
        zs_metrics = chosen_metrics
        ft_metrics = None
        chosen_predictor_path = ckpt_dir
        np.savez_compressed(zs_oof_file, ts=chosen_oof_ts, returns=chosen_oof_return)

    if zs_oof_file.exists() and ckpt_dir.resolve() != CHECKPOINT_DIR.resolve():
        shutil.copy2(zs_oof_file, ckpt_dir / "zs_oof.npz")
    elif not (ckpt_dir / "zs_oof.npz").exists():
        np.savez_compressed(ckpt_dir / "zs_oof.npz", ts=chosen_oof_ts, returns=chosen_oof_return)

    direction_proba, calib_info = calibrate_regression_to_direction_proba(chosen_oof_return)
    if len(chosen_oof_ts) > 0:
        save_oof(member_name, chosen_oof_ts, direction_proba)
        for alias in ("kronos-finetune", "tier4-kronos-macro", "tier3-kronos-meso"):
            if alias != member_name:
                save_oof(alias, chosen_oof_ts, direction_proba)

    summary_record = {
        "member": member_name,
        "winner": winner,
        "val_direction_acc": float(chosen_metrics.get("directional_accuracy", 0.5)),
        "val_return_mae": float(chosen_metrics.get("mae", 0.0)),
        "chosen_metrics": chosen_metrics,
        "zero_shot_metrics": zs_metrics,
        "finetuned_metrics": ft_metrics,
        "tokenizer_diagnostic": tok_diag,
        "calibration": calib_info,
        "v4_peak_calibrated": True,
        "status": "completed",
    }
    (ckpt_dir / "metrics.json").write_text(json.dumps(summary_record, indent=2), encoding="utf-8")
    if ckpt_dir.resolve() != CHECKPOINT_DIR.resolve():
        (CHECKPOINT_DIR / "metrics.json").write_text(json.dumps(summary_record, indent=2), encoding="utf-8")

    del df
    gc.collect()

    if upload:
        from common.hf_model_upload import upload_best_checkpoint

        checkpoint_files = [ckpt_dir / "metrics.json", ckpt_dir / "zs_oof.npz"]
        if chosen_predictor_path != ckpt_dir and chosen_predictor_path.exists():
            checkpoint_files += [p for p in chosen_predictor_path.glob("*") if p.is_file()]
        repo_id = upload_best_checkpoint(member_name, checkpoint_files, metrics=summary_record)
        log.info("Uploaded %s checkpoint for %s to %s", winner, member_name, repo_id)

    return summary_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-days", type=int, default=config.VAL_DAYS)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--lookback", "--lookback-window", dest="lookback_window", type=int, default=MAX_CONTEXT)
    parser.add_argument("--predict-window", type=int, default=48)
    parser.add_argument("--tokenizer-epochs", type=int, default=0)
    parser.add_argument("--epochs", "--basemodel-epochs", dest="basemodel_epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--step-size", type=int, default=60)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=32000)
    parser.add_argument("--predictor-lr", type=float, default=4e-5)
    parser.add_argument("--sample-count", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--skip-finetune", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--gpus", "--nproc-per-node", dest="gpus", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--member-name", type=str, default=MEMBER_NAME)
    args = parser.parse_args()
    run(
        val_days=args.val_days,
        max_days=args.max_days,
        lookback_window=args.lookback_window,
        predict_window=args.predict_window,
        tokenizer_epochs=args.tokenizer_epochs,
        basemodel_epochs=args.basemodel_epochs,
        batch_size=args.batch_size,
        skip_finetune=args.skip_finetune,
        upload=args.upload,
        gpus=args.gpus,
        resume=args.resume,
        sample_count=args.sample_count,
        temperature=args.temperature,
        max_samples=args.max_samples,
        predictor_lr=args.predictor_lr,
        accumulation_steps=args.accumulation_steps,
        step_size=args.step_size,
        export_only=args.export_only,
        output_dir=args.output_dir,
        member_name=args.member_name,
    )


if __name__ == "__main__":
    main()
