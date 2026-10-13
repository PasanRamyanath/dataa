"""Train Temporal Fusion Transformer (TFT) on btc_futures_features_10s*.parquet.

Designed for Kaggle GPU VM (2x T4):
- PyTorch Forecasting TFT + Lightning
- DDP multi-GPU when available
- Mixed precision
- Optimized DataLoader settings

Example (Kaggle notebook cell):

    !pip -q install pytorch-forecasting pytorch-lightning pyarrow fastparquet
    !python train_tft_kaggle.py --parquet /kaggle/input/your-dataset/btc_futures_features_10s_clean.parquet \
        --out-dir /kaggle/working/tft_out --max-epochs 30 --batch-size 64

"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch


def _bool_action(default: bool) -> argparse.Action:
    """Return a boolean flag action compatible with Kaggle Python.

    Uses BooleanOptionalAction when available so both --flag/--no-flag work.
    """

    action = getattr(argparse, "BooleanOptionalAction", None)
    if action is None:
        # Fallback: only supports enabling
        return "store_true"  # type: ignore[return-value]
    return action  # type: ignore[return-value]


def resolve_parquet_path(
    *,
    local_path: str | None,
    hf_data_repo_id: str,
    hf_data_filename: str,
    hf_data_repo_type: str,
    hf_token: str,
    hf_revision: str | None,
    hf_cache_dir: str | None,
    hf_force_download: bool,
) -> str:
    """Resolve parquet path either from local filesystem or by downloading from HuggingFace."""

    if local_path:
        return local_path

    if not hf_data_repo_id:
        raise ValueError("Provide --parquet or set --hf-data-repo-id to download from HuggingFace")

    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "huggingface_hub is required to download parquet from HF. Install in Kaggle: pip install -q huggingface_hub"
        ) from exc

    return hf_hub_download(
        repo_id=hf_data_repo_id,
        filename=hf_data_filename,
        repo_type=hf_data_repo_type,
        token=hf_token or None,
        revision=hf_revision,
        cache_dir=hf_cache_dir,
        force_download=hf_force_download,
    )


class HuggingFaceCheckpointUploader:
    """Upload checkpoints to HuggingFace Hub after each epoch.

    Notes:
    - Runs only on global rank 0 (DDP-safe).
    - Overwrites by reusing stable paths in the repo (e.g. checkpoints/best.ckpt).
    """

    def __init__(
        self,
        *,
        repo_id: str,
        token: str,
        path_in_repo_dir: str,
        upload_best: bool,
        upload_last: bool,
        overwrite: bool,
    ) -> None:
        self.repo_id = repo_id
        self.token = token
        self.path_in_repo_dir = path_in_repo_dir.strip("/")
        self.upload_best = upload_best
        self.upload_last = upload_last
        self.overwrite = overwrite

        self._api = None

    def setup(self, trainer, pl_module, stage: str | None = None) -> None:
        if not trainer.is_global_zero:
            return
        if not self.repo_id:
            raise ValueError("--hf-repo-id is required when --hf-upload is set")
        if not self.token:
            raise ValueError(
                "HF token missing. Set env var HF_TOKEN or pass --hf-token when --hf-upload is set."
            )

        try:
            from huggingface_hub import HfApi
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "huggingface_hub is required for uploads. Install it in Kaggle: pip install -q huggingface_hub"
            ) from exc

        self._api = HfApi(token=self.token)

    def _upload(self, *, local_path: str, filename: str, commit_message: str) -> None:
        if self._api is None:
            return
        if not os.path.exists(local_path):
            return

        # Stable destination path so each epoch can overwrite.
        if self.path_in_repo_dir:
            path_in_repo = f"{self.path_in_repo_dir}/{filename}"
        else:
            path_in_repo = filename

        # huggingface_hub overwrites by re-uploading to the same repo path.
        # `overwrite` is kept as a logical flag; the stable name ensures replacement.
        _ = self.overwrite

        self._api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo,
            repo_id=self.repo_id,
            repo_type="dataset",
            commit_message=commit_message,
        )

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if not trainer.is_global_zero:
            return

        # Find the ModelCheckpoint callback
        ckpt_cb = None
        for cb in trainer.callbacks:
            if cb.__class__.__name__ == "ModelCheckpoint":
                ckpt_cb = cb
                break
        if ckpt_cb is None:
            return

        epoch = int(trainer.current_epoch)
        commit_message = f"Upload checkpoints: epoch={epoch}"

        # Upload "best so far" and "last epoch" checkpoints with stable names.
        if self.upload_best:
            best_path = getattr(ckpt_cb, "best_model_path", "")
            # Prefer stable file if present
            stable_best = os.path.join(getattr(ckpt_cb, "dirpath", ""), "best.ckpt")
            self._upload(
                local_path=stable_best if os.path.exists(stable_best) else best_path,
                filename="best.ckpt",
                commit_message=commit_message,
            )

        if self.upload_last:
            last_path = getattr(ckpt_cb, "last_model_path", "")
            stable_last = os.path.join(getattr(ckpt_cb, "dirpath", ""), "last.ckpt")
            self._upload(
                local_path=stable_last if os.path.exists(stable_last) else last_path,
                filename="last.ckpt",
                commit_message=commit_message,
            )


@dataclass(frozen=True)
class DatasetSpec:
    time_col: str
    target_col: str
    group_col: str
    time_idx_col: str
    known_reals: List[str]
    unknown_reals: List[str]


KNOWN_TIME_FEATURE_CANDIDATES: Tuple[str, ...] = (
    # cyclical
    "hour_sin",
    "hour_cos",
    "minute_sin",
    "minute_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "day_of_month_sin",
    "day_of_month_cos",
    "month_sin",
    "month_cos",
    "quarter_sin",
    "quarter_cos",
    "week_of_year_sin",
    "week_of_year_cos",
    "is_month_start",
    # sessions
    "hour_0_8_asian",
    "hour_8_16_europe",
    "hour_16_24_us",
    "asian_open",
    "europe_open",
    "us_open",
    "us_close",
    "weekend",
    "weekday",
    "friday_close",
    # funding schedule
    "hours_to_funding",
    "minutes_to_funding",
    "hours_from_funding",
    "funding_window",
    "pre_funding_1h",
    "post_funding_1h",
    "funding_cycle_phase",
    "time_index_normalized",
    "relative_position",
    # calendar
    "is_holiday",
    "holiday_eve",
    "holiday_after",
    "month_end",
    "quarter_end",
)


def _best_effort_num_workers(requested: int | None) -> int:
    cpu = os.cpu_count() or 2
    if requested is None:
        # Kaggle benefits from parallel parquet decode + batch assembly.
        # Default: up to 12, leaving a bit of headroom.
        return max(2, min(12, max(1, cpu - 1)))
    return max(0, requested)


def _coerce_numeric_features_inplace(df: pd.DataFrame, cols: List[str]) -> None:
    """Coerce feature columns to numeric dtypes friendly to GPU training."""

    for c in cols:
        if pd.api.types.is_bool_dtype(df[c]):
            df[c] = df[c].astype(np.int8)
        elif pd.api.types.is_integer_dtype(df[c]):
            df[c] = df[c].astype(np.float32)
        elif pd.api.types.is_float_dtype(df[c]):
            # Prefer float32 for RAM/throughput.
            if df[c].dtype == np.float64:
                df[c] = df[c].astype(np.float32)


def _drop_constant_feature_columns(df: pd.DataFrame, cols: List[str]) -> List[str]:
    """Drop constant / all-NaN feature columns for stability and speed."""

    kept: List[str] = []
    for c in cols:
        if c not in df.columns:
            continue
        # Treat NaN as a separate value; nunique(dropna=False) catches all-NaN.
        nunique = int(df[c].nunique(dropna=False))
        if nunique <= 1:
            df.drop(columns=[c], inplace=True)
            continue
        kept.append(c)
    return kept


def _ensure_datetime(df: pd.DataFrame, col: str) -> None:
    if not np.issubdtype(df[col].dtype, np.datetime64):
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)


def _build_time_idx(df: pd.DataFrame, time_col: str, interval_seconds: int) -> pd.Series:
    # Use integer time index for TimeSeriesDataSet.
    t0 = df[time_col].min()
    delta = (df[time_col] - t0).dt.total_seconds()
    return (delta / float(interval_seconds)).round().astype(np.int64)


def infer_dataset_spec(df: pd.DataFrame, *, interval_seconds: int = 10) -> DatasetSpec:
    time_col = "timestamp" if "timestamp" in df.columns else None
    if time_col is None:
        raise ValueError("Expected a 'timestamp' column in the parquet.")

    target_col = "target" if "target" in df.columns else None
    if target_col is None:
        raise ValueError("Expected a 'target' column in the parquet.")

    group_col = "series_id"
    time_idx_col = "time_idx"

    # Ensure timestamp is usable
    _ensure_datetime(df, time_col)

    # Build a single-group id (single asset)
    df[group_col] = 0

    # Build time_idx
    df[time_idx_col] = _build_time_idx(df, time_col, interval_seconds)

    # Ensure sorted for correct slicing and reproducible splits
    df.sort_values([group_col, time_idx_col], inplace=True)

    # Identify numeric columns for covariates
    exclude = {time_col, target_col, group_col, time_idx_col}

    numeric_cols: List[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_bool_dtype(df[c]):
            numeric_cols.append(c)
        elif pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)

    known_reals = [c for c in KNOWN_TIME_FEATURE_CANDIDATES if c in df.columns]

    # If temporal features are missing (e.g., older parquet), generate a minimal set.
    if not known_reals:
        dt = df[time_col].dt.tz_convert(None)
        df["hour_sin"] = np.sin(2 * np.pi * (dt.dt.hour / 24.0)).astype(np.float32)
        df["hour_cos"] = np.cos(2 * np.pi * (dt.dt.hour / 24.0)).astype(np.float32)
        df["day_of_week_sin"] = np.sin(2 * np.pi * (dt.dt.dayofweek / 7.0)).astype(np.float32)
        df["day_of_week_cos"] = np.cos(2 * np.pi * (dt.dt.dayofweek / 7.0)).astype(np.float32)
        known_reals = ["hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos"]
        for c in known_reals:
            if c not in numeric_cols:
                numeric_cols.append(c)

    # All other numeric columns are treated as unknown reals (historical-only).
    unknown_reals = [c for c in numeric_cols if c not in known_reals]

    # Important: include the target as unknown real is not needed (handled separately).
    return DatasetSpec(
        time_col=time_col,
        target_col=target_col,
        group_col=group_col,
        time_idx_col=time_idx_col,
        known_reals=known_reals,
        unknown_reals=unknown_reals,
    )


def read_parquet_optimized(path: str) -> pd.DataFrame:
    # Keep float32 and avoid needless conversions.
    df = pd.read_parquet(path, engine="pyarrow")

    # Ensure float32 where possible to reduce RAM.
    float64_cols = df.select_dtypes(include=["float64"]).columns
    if len(float64_cols) > 0:
        df[float64_cols] = df[float64_cols].astype(np.float32)

    # Replace inf with NaN (should be rare if extractor already cleaned).
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", default=None, help="Path to features parquet (clean or raw). If omitted, downloads from HF.")
    parser.add_argument("--out-dir", default="tft_out", help="Output directory for checkpoints/logs")
    parser.add_argument("--interval-seconds", type=int, default=10)
    parser.add_argument("--encoder-len", type=int, default=720)
    parser.add_argument("--pred-len", type=int, default=90)
    parser.add_argument("--val-days", type=int, default=7, help="Validation window size (days from the end)")

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--attention-head-size", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gradient-clip-val", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--drop-constant-features",
        action=_bool_action(True),
        default=True,
        help="Drop constant/all-NaN feature columns to improve stability and speed",
    )

    parser.add_argument(
        "--target-normalizer",
        default="encoder",
        choices=["encoder", "group"],
        help=(
            "Target normalization strategy. 'encoder' usually works better for non-stationary series like prices; "
            "'group' is the classic per-series normalization."
        ),
    )

    parser.add_argument(
        "--torch-compile",
        action=_bool_action(False),
        default=False,
        help="Optional: torch.compile() the model for extra speed (PyTorch>=2).",
    )

    parser.add_argument(
        "--probabilistic",
        action=_bool_action(True),
        default=True,
        help=(
            "Train probabilistic (quantile) forecasts. "
            "Point-accuracy metrics (MAE/RMSE) are still logged using the median (0.5) quantile."
        ),
    )
    parser.add_argument(
        "--quantiles",
        default="0.1,0.5,0.9",
        help="Comma-separated quantiles used when --probabilistic is enabled (e.g. 0.05,0.5,0.95)",
    )

    # HuggingFace dataset download (optional)
    parser.add_argument(
        "--hf-data-repo-id",
        default=os.environ.get("HF_DATA_REPO_ID", os.environ.get("HF_REPO_ID", "")),
        help="HF dataset repo id hosting the parquet (defaults to HF_DATA_REPO_ID or HF_REPO_ID)",
    )
    parser.add_argument(
        "--hf-data-filename",
        default=os.environ.get("HF_DATA_FILENAME", "btc_futures_features_10s_clean.parquet"),
        help="Filename/path in repo for parquet (default: btc_futures_features_10s_clean.parquet)",
    )
    parser.add_argument(
        "--hf-data-repo-type",
        default=os.environ.get("HF_DATA_REPO_TYPE", "dataset"),
        choices=["dataset", "model", "space"],
        help="HF repo type containing the parquet",
    )
    parser.add_argument(
        "--hf-data-revision",
        default=os.environ.get("HF_DATA_REVISION", None),
        help="Optional HF revision (branch/tag/commit) for parquet download",
    )
    parser.add_argument(
        "--hf-cache-dir",
        default=os.environ.get("HF_HOME", None),
        help="Optional cache dir for HF downloads (defaults to HF_HOME if set)",
    )
    parser.add_argument(
        "--hf-force-download",
        action=_bool_action(False),
        default=False,
        help="Force re-download parquet even if cached",
    )

    # HuggingFace uploads (optional)
    parser.add_argument("--hf-upload", action="store_true", help="Upload best/last checkpoints each epoch")
    parser.add_argument(
        "--hf-repo-id",
        default=os.environ.get("HF_REPO_ID", ""),
        help="HuggingFace dataset repo id, e.g. p-j-r-1-2-3/Project-IMPJR-ModelTFT-dataset",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN", "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"),
        help="HuggingFace token (hardcoded default; or set env var HF_TOKEN)",
    )
    parser.add_argument(
        "--hf-path",
        default="tft_checkpoints",
        help="Directory inside the repo to place checkpoints (stable names)",
    )
    parser.add_argument(
        "--hf-upload-best",
        action=_bool_action(True),
        default=True,
        help="Upload best.ckpt each epoch",
    )
    parser.add_argument(
        "--hf-upload-last",
        action=_bool_action(True),
        default=True,
        help="Upload last.ckpt each epoch",
    )
    parser.add_argument(
        "--hf-overwrite",
        action=_bool_action(True),
        default=True,
        help="Overwrite checkpoints in repo by using stable names",
    )

    parser.add_argument("--precision", default="16-mixed", choices=["16-mixed", "32-true"], help="AMP setting")
    # Use parse_known_args so the script can be launched from Jupyter/Colab
    # where the kernel injects extra flags (e.g. '-f <json>') which argparse
    # would otherwise treat as errors.
    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        print(f"Note: Ignoring unknown CLI args: {unknown_args[:5]}{'...' if len(unknown_args)>5 else ''}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    parquet_path = resolve_parquet_path(
        local_path=args.parquet,
        hf_data_repo_id=args.hf_data_repo_id,
        hf_data_filename=args.hf_data_filename,
        hf_data_repo_type=args.hf_data_repo_type,
        hf_token=args.hf_token,
        hf_revision=args.hf_data_revision,
        hf_cache_dir=args.hf_cache_dir,
        hf_force_download=args.hf_force_download,
    )

    # -------- Read data --------
    df = read_parquet_optimized(parquet_path)

    # -------- Build dataset spec --------
    spec = infer_dataset_spec(df, interval_seconds=args.interval_seconds)

    # Coerce feature dtypes for speed and memory.
    _coerce_numeric_features_inplace(df, spec.known_reals + spec.unknown_reals)

    if args.drop_constant_features:
        spec = DatasetSpec(
            time_col=spec.time_col,
            target_col=spec.target_col,
            group_col=spec.group_col,
            time_idx_col=spec.time_idx_col,
            known_reals=_drop_constant_feature_columns(df, spec.known_reals),
            unknown_reals=_drop_constant_feature_columns(df, spec.unknown_reals),
        )

    # Ensure we are using all numeric feature columns as covariates.
    total_covariates = len(spec.known_reals) + len(spec.unknown_reals)
    if total_covariates <= 0:
        raise ValueError("No covariate columns detected. Check parquet schema.")

    # -------- Train/val split --------
    steps_per_day = int((24 * 60 * 60) / args.interval_seconds)
    val_steps = args.val_days * steps_per_day

    max_time_idx = int(df[spec.time_idx_col].max())
    min_prediction_idx = max(0, max_time_idx - val_steps)

    # Import here so this script can be inspected without installed deps
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger

    # Faster + safer DDP defaults on Kaggle dual-T4
    DDPStrategy = None
    try:
        from pytorch_lightning.strategies import DDPStrategy as _DDPStrategy

        DDPStrategy = _DDPStrategy
    except Exception:
        DDPStrategy = None

    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.data.encoders import EncoderNormalizer
    from pytorch_forecasting.metrics import MAE, RMSE, QuantileLoss
    from pytorch_forecasting.models import TemporalFusionTransformer

    # Build training dataset only from history strictly before validation window.
    train_df = df[df[spec.time_idx_col] < min_prediction_idx].copy()

    if len(train_df) < (args.encoder_len + args.pred_len + 1):
        raise ValueError(
            "Not enough training rows after split. "
            "Try reducing --val-days or using a longer parquet."
        )

    if args.target_normalizer == "encoder":
        target_normalizer = EncoderNormalizer()
    else:
        target_normalizer = GroupNormalizer(groups=[spec.group_col])

    training = TimeSeriesDataSet(
        train_df,
        time_idx=spec.time_idx_col,
        target=spec.target_col,
        group_ids=[spec.group_col],
        min_encoder_length=args.encoder_len,
        max_encoder_length=args.encoder_len,
        min_prediction_length=args.pred_len,
        max_prediction_length=args.pred_len,
        time_varying_known_reals=[spec.time_idx_col] + spec.known_reals,
        time_varying_unknown_reals=spec.unknown_reals,
        target_normalizer=target_normalizer,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    # Validation dataset: same transforms, predict starting at min_prediction_idx
    validation = TimeSeriesDataSet.from_dataset(
        training,
        df,
        min_prediction_idx=min_prediction_idx,
        stop_randomization=True,
    )

    num_workers = _best_effort_num_workers(args.num_workers)

    train_loader = training.to_dataloader(
        train=True,
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )

    val_loader = validation.to_dataloader(
        train=False,
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )

    # -------- Model --------
    if args.probabilistic:
        try:
            quantiles = [float(q.strip()) for q in str(args.quantiles).split(",") if q.strip()]
        except Exception as exc:
            raise ValueError(f"Invalid --quantiles value: {args.quantiles}") from exc

        if not quantiles:
            raise ValueError("--quantiles must contain at least one value")
        if 0.5 not in quantiles:
            raise ValueError("--quantiles must include 0.5 so point metrics can be computed from median")

        output_size = len(quantiles)
        loss = QuantileLoss(quantiles=quantiles)
    else:
        # Pure point-forecast training
        output_size = 1
        loss = RMSE()

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        attention_head_size=args.attention_head_size,
        dropout=args.dropout,
        hidden_continuous_size=args.hidden_size,
        output_size=output_size,
        loss=loss,
        logging_metrics=[MAE(), RMSE()],
        log_interval=50,
        reduce_on_plateau_patience=4,
    )

    if args.torch_compile:
        # Optional speed-up. Can be sensitive to versions; keep best-effort.
        try:
            tft = torch.compile(tft, mode="max-autotune")
        except Exception as exc:
            print(f"torch.compile failed; continuing without compile: {exc}")

    # -------- Trainer (multi-GPU on Kaggle when available) --------
    gpus = torch.cuda.device_count()
    accelerator = "gpu" if gpus > 0 else "cpu"
    devices = gpus if gpus > 0 else 1

    if gpus > 1 and DDPStrategy is not None:
        # Faster than generic 'ddp' because we avoid unused-parameter discovery.
        strategy = DDPStrategy(find_unused_parameters=False)
    else:
        strategy = "ddp" if gpus > 1 else "auto"

    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir / "checkpoints"),
        filename="best",
        auto_insert_metric_name=False,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        every_n_epochs=1,
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=6,
        min_delta=1e-4,
    )

    logger = CSVLogger(save_dir=str(out_dir), name="logs")

    callbacks = [ckpt_cb, early_stop, LearningRateMonitor(logging_interval="step")]
    if args.hf_upload:
        callbacks.append(
            HuggingFaceCheckpointUploader(
                repo_id=args.hf_repo_id,
                token=args.hf_token,
                path_in_repo_dir=args.hf_path,
                upload_best=args.hf_upload_best,
                upload_last=args.hf_upload_last,
                overwrite=args.hf_overwrite,
            )
        )

    trainer = Trainer(
        default_root_dir=str(out_dir),
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=callbacks,
        logger=logger,
        enable_checkpointing=True,
        log_every_n_steps=20,
        num_sanity_val_steps=0,
        enable_model_summary=False,
        deterministic=False,
    )

    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Save best checkpoint path for convenience
    best_path = ckpt_cb.best_model_path
    last_path = ckpt_cb.last_model_path
    (out_dir / "best_checkpoint.txt").write_text(best_path)
    (out_dir / "last_checkpoint.txt").write_text(last_path)

    print(f"\nDone. Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")
    print(f"Known reals used: {len(spec.known_reals)}")
    print(f"Unknown reals used: {len(spec.unknown_reals)}")
    print(f"Total covariates used: {total_covariates}")
    if args.probabilistic:
        print("Point metrics (MAE/RMSE) are logged from the median (0.5) quantile.")
    print(f"Devices: {devices} ({accelerator}), strategy={strategy}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
