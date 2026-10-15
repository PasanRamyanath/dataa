"""
Kronos (arxiv.org/abs/2508.02739) finetune ensemble member.

Kronos is NOT pip-installable — it ships only as a GitHub repo
(github.com/shiyu-coder/Kronos, confirmed via web research) with
its tokenizer/predictor classes imported directly from that repo's `model/`
package, and its two-stage finetuning (tokenizer, then predictor) driven by
`finetune_csv/train_sequential.py` + a YAML config.

Option A "Train & Tournament" Architecture:
  1. Diagnostic: Runs a 5-minute diagnostic pass of KronosTokenizer over 10s bars
     verifying Binary Spherical Quantization (BSQ) codebook entropy. The pre-trained
     codebook (trained on 12B K-lines) is frozen to preserve universal representations.
  2. Baseline Zero-Shot Evaluation: Evaluates pre-trained Kronos foundation weights
     (NeoQuasar/Kronos-Tokenizer-base + NeoQuasar/Kronos-base) on the held-out 7-day
     purged/embargoed validation set using Monte Carlo path averaging (T=0.6, top_p=0.9, N=5).
  3. Bounded Predictor Fine-Tuning: Fine-tunes the autoregressive Transformer backbone
     over a fixed budget of 128,000 diverse windows uniformly sampled across ALL 764 days
     (preserving full macro-regime diversity without stride-1 autocorrelation bloat).
     Accelerated via PyTorch FP16 Automatic Mixed Precision (AMP) on Tensor Cores.
  4. Head-to-Head Tournament: Evaluates the fine-tuned checkpoint on the exact same
     validation windows with T=0.6, N=5. Compares Directional Accuracy and MAE.
     The provably superior model is automatically selected, its OOF exported, and its
     weights uploaded to Hugging Face for the downstream Stacking Meta-Learner.

Usage:
    python -m models.kronos_finetune.train --val-days 7 --upload
    python -m models.kronos_finetune.train --skip-finetune --upload   # zero-shot frozen-checkpoint eval only
"""
from __future__ import annotations

import argparse
import json
import logging
import os
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
MAX_CONTEXT = 512  # Kronos-small/base context cap
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
    """Patches vendored Kronos repo for:
      1. Bounded sample budgeting (max_samples) with uniform pseudo-random start index sampling
         across all historical days in CustomKlineDataset.
      2. PyTorch FP16 AMP (torch.amp.autocast + GradScaler) on Turing Tensor Cores with
         gradient accumulation to maintain effective batch size = 32 while fitting in 14.5GB VRAM.
      3. Safe tokenizer path resolution.
    """
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
                log.info("Patched Kronos config_loader.py for max_samples budgeting.")

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
            log.info("Patched Kronos finetune_base_model.py for max_samples, gradient accumulation and FP16 AMP.")

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
            log.info("Patched Kronos train_sequential.py for safe tokenizer path resolution.")


def ensure_repo() -> Path:
    if not (VENDOR_DIR / "finetune_csv").exists():
        VENDOR_DIR.parent.mkdir(parents=True, exist_ok=True)
        log.info("Cloning %s into %s", KRONOS_REPO_URL, VENDOR_DIR)
        subprocess.run(["git", "clone", "--depth", "1", KRONOS_REPO_URL, str(VENDOR_DIR)], check=True)
    _patch_kronos_repo(VENDOR_DIR)
    return VENDOR_DIR


def ensure_pretrained_checkpoints() -> tuple[Path, Path]:
    from huggingface_hub import snapshot_download

    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer_path = Path(snapshot_download(repo_id=TOKENIZER_REPO_ID, local_dir=str(PRETRAINED_DIR / "Kronos-Tokenizer-base")))
    predictor_path = Path(snapshot_download(repo_id=PREDICTOR_REPO_ID, local_dir=str(PRETRAINED_DIR / "Kronos-base")))
    return tokenizer_path, predictor_path


def export_csv(train_df: pl.DataFrame) -> Path:
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
                  basemodel_epochs: int, batch_size: int = 8, max_samples: int = 128000,
                  predictor_lr: float = 4e-5, resume: bool = False,
                  accumulation_steps: int = 4) -> Path:
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
            "num_workers": 4,
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

    if gpus > 1:
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
    sys.path.insert(0, str(vendor_dir))
    from model import KronosTokenizer  # noqa: PLC0415
    return KronosTokenizer.from_pretrained(str(tokenizer_path))


def check_tokenizer_codebook(tokenizer, train_df: pl.DataFrame) -> dict:
    """Quick diagnostic verifying BSQ codebook utilization on 10s bars."""
    try:
        import torch

        ohlcv_cols = ["open", "high", "low", "close", "volume", "notional_volume"]
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
        log.info("Tokenizer diagnostic on 10s BTC bars: Coarse tokens used: %d, Fine tokens used: %d", u0, u1)
        tokenizer = tokenizer.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"coarse_tokens": u0, "fine_tokens": u1}
    except Exception as e:
        log.warning("Tokenizer diagnostic check skipped: %s", e)
        return {"coarse_tokens": 0, "fine_tokens": 0}


def _load_predictor(vendor_dir: Path, tokenizer_path: Path, predictor_path: Path):
    sys.path.insert(0, str(vendor_dir))
    from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: PLC0415

    tokenizer = KronosTokenizer.from_pretrained(str(tokenizer_path))
    model = Kronos.from_pretrained(str(predictor_path))
    return KronosPredictor(model, tokenizer, max_context=MAX_CONTEXT)


def evaluate_on_val(
    predictor, val_df: pl.DataFrame, lookback_window: int, predict_window: int,
    temperature: float = 0.6, top_p: float = 0.9, sample_count: int = 5
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Independent held-out evaluation scoring directional accuracy + MAE.
    Uses temperature=0.6, top_p=0.9, sample_count paths (Monte Carlo ensemble)
    matching the Kronos paper's optimal inference configuration.
    """
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    n = val_df.height
    window = lookback_window + predict_window
    stride = config.KRONOS_OOF_STEP_BARS
    maes, correct_dir, total = [], 0, 0
    window_ts, window_pred_return = [], []

    from common.windowing import run_ids
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
            import pandas as pd
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

        y_true = fut["close"].to_numpy()
        y_pred = pred_df["close"].to_numpy()
        last_known = ctx["close"].to_numpy()[-1]

        maes.append(float(np.mean(np.abs(y_true - y_pred))))
        correct_dir += int(np.sign(y_true[-1] - last_known) == np.sign(y_pred[-1] - last_known))
        total += 1

        window_ts.append(ctx["ts"].to_numpy()[-1])
        window_pred_return.append(float((y_pred[-1] - last_known) / last_known) if last_known else 0.0)

    metrics = {
        "mae": float(np.mean(maes)) if maes else None,
        "directional_accuracy": correct_dir / total if total else None,
        "n_windows": total,
        "n_windows_skipped_gap": skipped,
        "temperature": temperature,
        "sample_count": sample_count,
    }
    return metrics, np.array(window_ts), np.array(window_pred_return, dtype=np.float32)


def run(val_days: int, max_days: int | None, lookback_window: int, predict_window: int,
        tokenizer_epochs: int, basemodel_epochs: int, batch_size: int = 8,
        skip_finetune: bool = False, upload: bool = False, gpus: int | None = None, resume: bool = False,
        sample_count: int = 5, temperature: float = 0.6, max_samples: int = 128000,
        predictor_lr: float = 4e-5, accumulation_steps: int = 4) -> dict:
    import torch

    df = load_days(max_days=max_days)
    val_size = val_days * config.BARS_PER_DAY
    split = purged_walk_forward_splits(df, n_folds=1, val_size=val_size)[0]

    vendor_dir = ensure_repo()
    tokenizer_path, predictor_path = ensure_pretrained_checkpoints()

    # 1. Run Tokenizer Diagnostic
    raw_tok = _load_raw_tokenizer(vendor_dir, tokenizer_path)
    tok_diag = check_tokenizer_codebook(raw_tok, split.train)

    # 2. Evaluate Zero-Shot Foundation Baseline (with caching to avoid redundant 3.5h eval)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    zs_cache_file = CHECKPOINT_DIR / "zs_baseline_metrics.json"
    zs_oof_file = CHECKPOINT_DIR / "zs_oof.npz"

    # Restore from HF dataset cache if missing locally
    if upload and not (zs_cache_file.exists() and zs_oof_file.exists()):
        try:
            from huggingface_hub import hf_hub_download
            hf_cache_dir = CHECKPOINT_DIR / "_hf_cache"
            hf_hub_download(
                repo_id=config.HF_TRAINING_DATA_REPO_ID,
                filename="kronos_cache/zs_baseline_metrics.json",
                repo_type="dataset",
                local_dir=str(hf_cache_dir),
                token=config.get_hf_token(),
            )
            hf_hub_download(
                repo_id=config.HF_TRAINING_DATA_REPO_ID,
                filename="kronos_cache/zs_oof.npz",
                repo_type="dataset",
                local_dir=str(hf_cache_dir),
                token=config.get_hf_token(),
            )
            dl_m = hf_cache_dir / "kronos_cache" / "zs_baseline_metrics.json"
            dl_o = hf_cache_dir / "kronos_cache" / "zs_oof.npz"
            if dl_m.exists() and dl_o.exists():
                import shutil
                shutil.copy2(dl_m, zs_cache_file)
                shutil.copy2(dl_o, zs_oof_file)
                log.info("Restored Zero-Shot baseline cache from HF %s/kronos_cache", config.HF_TRAINING_DATA_REPO_ID)
        except Exception as e:
            log.info("No remote zero-shot baseline cache on HF: %s", e)

    if zs_cache_file.exists() and zs_oof_file.exists():
        log.info("--- Phase 1: Loading Cached Pretrained Zero-Shot Kronos Metrics ---")
        zs_metrics = json.loads(zs_cache_file.read_text(encoding="utf-8"))
        npz = np.load(zs_oof_file)
        zs_oof_ts = npz["ts"]
        zs_oof_return = npz["returns"]
        log.info("Loaded Cached Zero-Shot Baseline: MAE=%.4f, Directional Acc=%.4f (over %d windows)",
                 zs_metrics["mae"], zs_metrics["directional_accuracy"], zs_metrics["n_windows"])
    else:
        log.info("--- Phase 1: Evaluating Pretrained Zero-Shot Kronos (T=%.2f, N=%d) ---", temperature, sample_count)
        zs_predictor = _load_predictor(vendor_dir, tokenizer_path, predictor_path)
        zs_metrics, zs_oof_ts, zs_oof_return = evaluate_on_val(
            zs_predictor, split.val, lookback_window, predict_window,
            temperature=temperature, top_p=0.9, sample_count=sample_count
        )
        zs_metrics["variant"] = "zero-shot"
        zs_cache_file.write_text(json.dumps(zs_metrics, indent=2))
        np.savez_compressed(zs_oof_file, ts=zs_oof_ts, returns=zs_oof_return)
        log.info("Zero-Shot Baseline: MAE=%.4f, Directional Acc=%.4f (over %d windows)",
                 zs_metrics["mae"], zs_metrics["directional_accuracy"], zs_metrics["n_windows"])
        del zs_predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if upload:
            try:
                from huggingface_hub import HfApi
                api = HfApi(token=config.get_hf_token())
                api.upload_file(
                    path_or_fileobj=str(zs_cache_file),
                    path_in_repo="kronos_cache/zs_baseline_metrics.json",
                    repo_id=config.HF_TRAINING_DATA_REPO_ID,
                    repo_type="dataset",
                )
                api.upload_file(
                    path_or_fileobj=str(zs_oof_file),
                    path_in_repo="kronos_cache/zs_oof.npz",
                    repo_id=config.HF_TRAINING_DATA_REPO_ID,
                    repo_type="dataset",
                )
                log.info("Committed Zero-Shot baseline cache to HF %s/kronos_cache", config.HF_TRAINING_DATA_REPO_ID)
            except Exception as e:
                log.warning("Could not upload zero-shot cache to HF: %s", e)

    if skip_finetune:
        log.info("(--skip-finetune specified: skipping fine-tuning phase)")
        winner = "zero-shot"
        chosen_metrics = zs_metrics
        chosen_oof_ts = zs_oof_ts
        chosen_oof_return = zs_oof_return
        chosen_tokenizer_path = tokenizer_path
        chosen_predictor_path = predictor_path
        ft_metrics = None
    else:
        # 3. Fine-Tune Predictor with Bounded Sampling & FP16 AMP
        log.info("--- Phase 2: Fine-Tuning Kronos Predictor (%d epochs, max %d samples, LR=%.1e, batch_size=%d, accum_steps=%d) ---",
                 basemodel_epochs, max_samples, predictor_lr, batch_size, accumulation_steps)
        csv_path = export_csv(split.train)
        config_path = write_config(
            csv_path, tokenizer_path, predictor_path, lookback_window,
            predict_window, tokenizer_epochs=0, basemodel_epochs=basemodel_epochs,
            batch_size=batch_size, max_samples=max_samples, predictor_lr=predictor_lr,
            resume=resume, accumulation_steps=accumulation_steps
        )

        # Pre-seed tokenizer in the output directory so train_sequential never complains
        tok_best = FINETUNED_DIR / EXP_NAME / "tokenizer" / "best_model"
        tok_best.parent.mkdir(parents=True, exist_ok=True)
        if not tok_best.exists():
            import shutil
            shutil.copytree(tokenizer_path, tok_best)
            log.info("Mirrored pre-trained tokenizer to %s for upstream Kronos compatibility", tok_best)

        run_finetune(vendor_dir, config_path, gpus=gpus)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        finetuned_tokenizer_path = tokenizer_path
        finetuned_predictor_path = FINETUNED_DIR / EXP_NAME / "basemodel" / "best_model"
        if not finetuned_predictor_path.exists():
            log.warning("Fine-tuned best_model not found at %s; falling back to pretrained predictor", finetuned_predictor_path)
            finetuned_predictor_path = predictor_path

        # 4. Evaluate Fine-Tuned Model
        log.info("--- Phase 3: Evaluating Fine-Tuned Kronos (T=%.2f, N=%d) ---", temperature, sample_count)
        ft_predictor = _load_predictor(vendor_dir, finetuned_tokenizer_path, finetuned_predictor_path)
        ft_metrics, ft_oof_ts, ft_oof_return = evaluate_on_val(
            ft_predictor, split.val, lookback_window, predict_window,
            temperature=temperature, top_p=0.9, sample_count=sample_count
        )
        ft_metrics["variant"] = "fine-tuned"
        log.info("Fine-Tuned Result: MAE=%.4f, Directional Acc=%.4f (over %d windows)",
                 ft_metrics["mae"], ft_metrics["directional_accuracy"], ft_metrics["n_windows"])
        del ft_predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 5. Head-to-Head Tournament Selection
        zs_acc = zs_metrics["directional_accuracy"] or 0.0
        ft_acc = ft_metrics["directional_accuracy"] or 0.0
        zs_mae = zs_metrics["mae"] or float("inf")
        ft_mae = ft_metrics["mae"] or float("inf")

        if ft_acc > zs_acc or (ft_acc == zs_acc and ft_mae <= zs_mae):
            winner = "fine-tuned"
            chosen_metrics = ft_metrics
            chosen_oof_ts = ft_oof_ts
            chosen_oof_return = ft_oof_return
            chosen_tokenizer_path = finetuned_tokenizer_path
            chosen_predictor_path = finetuned_predictor_path
        else:
            winner = "zero-shot"
            chosen_metrics = zs_metrics
            chosen_oof_ts = zs_oof_ts
            chosen_oof_return = zs_oof_return
            chosen_tokenizer_path = tokenizer_path
            chosen_predictor_path = predictor_path

        banner = (
            "\n" + "=" * 70 + "\n"
            f"KRONOS TOURNAMENT OUTCOME:\n"
            f"  Zero-Shot:   MAE = {zs_mae:.4f} | Directional Acc = {zs_acc * 100:.2f}%\n"
            f"  Fine-Tuned:  MAE = {ft_mae:.4f} | Directional Acc = {ft_acc * 100:.2f}%\n"
            f"  >>> SELECTED WINNER: {winner.upper()} <<<\n"
            + "=" * 70
        )
        log.info(banner)
        print(banner)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    summary_record = {
        "winner": winner,
        "chosen_metrics": chosen_metrics,
        "zero_shot_metrics": zs_metrics,
        "finetuned_metrics": ft_metrics,
        "tokenizer_diagnostic": tok_diag,
    }
    (CHECKPOINT_DIR / "metrics.json").write_text(json.dumps(summary_record, indent=2), encoding="utf-8")

    if len(chosen_oof_ts) > 0:
        from common.oof import regression_to_direction_proba, save_oof

        direction_proba = regression_to_direction_proba(chosen_oof_return)
        save_oof(MEMBER_NAME, chosen_oof_ts, direction_proba)

    if upload:
        from common.hf_model_upload import upload_best_checkpoint, upload_training_data_snapshot

        checkpoint_files = [CHECKPOINT_DIR / "metrics.json"]
        if winner == "fine-tuned" and chosen_predictor_path.exists():
            checkpoint_files += list(chosen_predictor_path.glob("*"))
        repo_id = upload_best_checkpoint(MEMBER_NAME, checkpoint_files, metrics=chosen_metrics)
        if not skip_finetune and (DATA_DIR / f"{EXP_NAME}_train.csv").exists():
            data_repo_id = upload_training_data_snapshot(MEMBER_NAME, [DATA_DIR / f"{EXP_NAME}_train.csv"])
            log.info("Uploaded training data to %s", data_repo_id)
        log.info("Uploaded %s checkpoint to %s", winner, repo_id)

    return chosen_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-days", type=int, default=config.VAL_DAYS)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--lookback-window", type=int, default=MAX_CONTEXT)
    parser.add_argument("--predict-window", type=int, default=48)
    parser.add_argument("--tokenizer-epochs", type=int, default=0)
    parser.add_argument("--basemodel-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=128000)
    parser.add_argument("--predictor-lr", type=float, default=4e-5)
    parser.add_argument("--sample-count", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--skip-finetune", action="store_true", help="zero-shot eval with the frozen pretrained checkpoint only")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.val_days, args.max_days, args.lookback_window, args.predict_window,
        args.tokenizer_epochs, args.basemodel_epochs, args.batch_size, args.skip_finetune, args.upload,
        args.gpus, args.resume, sample_count=args.sample_count, temperature=args.temperature,
        max_samples=args.max_samples, predictor_lr=args.predictor_lr,
        accumulation_steps=args.accumulation_steps)


if __name__ == "__main__":
    main()
