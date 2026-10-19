"""
iTransformer ensemble member, via neuralforecast's iTransformer
implementation — see common/neuralforecast_harness.py for the shared
data-prep/fit/eval loop every neuralforecast-based member (tft, patchtst,
this, nhits) uses.

iTransformer's constructor requires `n_series` (it's designed for
multivariate panels of series attending over each other) — we only have one
series (BTCUSDT), so n_series=1; it still benefits from the architecture's
inverted-attention treatment of the exogenous feature channels themselves.

Usage:
    python -m models.itransformer.train --val-days 7 --max-steps 1000 --upload
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
log = logging.getLogger("models.itransformer.train")

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
MEMBER_NAME = "itransformer"


def run(val_days: int, max_days: int | None, max_steps: int, upload: bool, gpus: int | None = None) -> dict:
    from neuralforecast.models import iTransformer

    from common.neuralforecast_harness import run_member

    df = load_days(max_days=max_days)  # unfiltered: see common/neuralforecast_harness.py
    val_size = val_days * config.BARS_PER_DAY
    split = purged_walk_forward_splits(df, n_folds=1, val_size=val_size)[0]

    result = run_member(
        MEMBER_NAME, iTransformer, df, split.fold,
        max_steps=max_steps, checkpoint_dir=CHECKPOINT_DIR, gpus=gpus,
        extra_model_kwargs={"n_series": 1, "learning_rate": 1e-4, "windows_batch_size": 16, "batch_size": 16, "inference_windows_batch_size": 16},
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
