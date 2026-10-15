"""
N-HiTS ensemble member, via neuralforecast's NHITS implementation — see
common/neuralforecast_harness.py for the shared data-prep/fit/eval loop
every neuralforecast-based member (tft, patchtst, itransformer, this) uses.

Usage:
    python -m models.nhits.train --val-days 7 --max-steps 1000 --upload
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import config
from common.cv import purged_walk_forward_splits
from common.data import load_days

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("models.nhits.train")

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
MEMBER_NAME = "nhits"


# Pruned continuous high-conviction features: avoids feeding 184 raw exogenous columns
# into unconstrained MLPs without a Variable Selection Network, and standard scaling
# eliminates division-by-zero on sparse features with zero IQR under RobustScaler.
NHITS_PRUNED_HIST_EXOG = [
    "realized_vol_30", "realized_vol_90", "realized_vol_360",
    "order_flow_imbalance_near", "order_flow_imbalance_wide",
    "ofi_imbalance_decay_rate", "ofi_imbalance_decay_slope",
    "taker_buy_sell_imbalance", "volume_delta",
    "log_return_1", "log_return_6", "log_return_12",
    "proxy_spread_pct", "roll_spread_30",
]
NHITS_PRUNED_FUTR_EXOG = [
    "hour_sin", "hour_cos", "minute_sin", "minute_cos", "funding_cycle_phase",
]


def run(val_days: int, max_days: int | None, max_steps: int, upload: bool, gpus: int | None = None) -> dict:
    from neuralforecast.models import NHITS

    from common.neuralforecast_harness import run_member

    df = load_days(max_days=max_days)  # unfiltered: see common/neuralforecast_harness.py
    val_size = val_days * config.BARS_PER_DAY
    split = purged_walk_forward_splits(df, n_folds=1, val_size=val_size)[0]

    scaler = getattr(config, "NHITS_SCALER_TYPE", "standard")
    result = run_member(
        MEMBER_NAME, NHITS, df, split.fold,
        max_steps=max_steps, checkpoint_dir=CHECKPOINT_DIR, gpus=gpus,
        extra_model_kwargs={
            "learning_rate": 3e-4,
            "windows_batch_size": 16,
            "batch_size": 16,
            "inference_windows_batch_size": 16,
            "scaler_type": scaler,
            "interpolation_mode": "linear",
            "hist_exog_list": NHITS_PRUNED_HIST_EXOG,
            "futr_exog_list": NHITS_PRUNED_FUTR_EXOG,
        },
    )

    training_data_path = CHECKPOINT_DIR / "training_data_snapshot.parquet"
    split.train.write_parquet(training_data_path, compression="snappy")
    (CHECKPOINT_DIR / "metrics.json").write_text(json.dumps(result["metrics"], indent=2), encoding="utf-8")
    log.info("Saved checkpoint to %s", CHECKPOINT_DIR)

    if upload:
        from common.hf_model_upload import upload_best_checkpoint, upload_training_data_snapshot

        checkpoint_files = list(CHECKPOINT_DIR.glob("*"))
        repo_id = upload_best_checkpoint(MEMBER_NAME, checkpoint_files, metrics=result["metrics"])
        data_repo_id = upload_training_data_snapshot(MEMBER_NAME, [training_data_path])
        log.info("Uploaded checkpoint to %s and training data to %s", repo_id, data_repo_id)

    return result["metrics"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-days", type=int, default=config.VAL_DAYS)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--gpus", type=int, default=None, help="GPU count for training; default auto-detects via torch.cuda.device_count()")
    args = parser.parse_args()
    run(args.val_days, args.max_days, args.max_steps, args.upload, args.gpus)


if __name__ == "__main__":
    main()
