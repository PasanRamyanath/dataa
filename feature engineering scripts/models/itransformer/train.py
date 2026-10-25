"""
iTransformer ensemble member (`itransformer` / `tier3-itransformer-32` / `tier3-itransformer-meso`):
inverted-dimension transformer via neuralforecast with robust window scaling
and bounded T4 batch sizes.
"""
from __future__ import annotations

import logging
from pathlib import Path

import config
from common.neuralforecast_harness import (
    make_arg_parser,
    run_member,
    trainer_Accelerator_kwargs,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
MEMBER_NAME = "itransformer"
DEFAULT_H = config.FORECAST_PERIODS


def build_model(input_size: int, h: int, max_steps: int, gpus: int | None = None):
    from neuralforecast.models import iTransformer

    return iTransformer(
        h=h,
        input_size=input_size,
        n_series=1,
        hidden_size=128,
        n_heads=8,
        e_layers=3,
        d_ff=256,
        dropout=0.12,
        learning_rate=1e-4,
        max_steps=max_steps,
        batch_size=16,
        windows_batch_size=16,
        inference_windows_batch_size=16,
        scaler_type="robust",
        random_seed=42,
        **trainer_Accelerator_kwargs(gpus),
    )


def run(
    val_days: int = config.VAL_DAYS,
    max_days: int | None = None,
    max_steps: int = 2500,
    upload: bool = False,
    gpus: int | None = None,
    input_size: int = config.LOOKBACK_PERIODS,
    h: int = DEFAULT_H,
    step_size: int = config.SEQ_OOF_STEP_BARS,
    output_dir: str | Path | None = None,
    member_name: str = MEMBER_NAME,
) -> dict:
    return run_member(
        member_name=member_name or MEMBER_NAME,
        model_col="iTransformer",
        checkpoint_dir=CHECKPOINT_DIR,
        build_model=lambda inp, hor, steps: build_model(inp, hor, steps, gpus),
        exog_cols=[],
        val_days=val_days,
        max_days=max_days,
        input_size=input_size,
        h=h,
        max_steps=max_steps,
        step_size=step_size,
        upload=upload,
        output_dir=output_dir,
        oof_aliases=["itransformer", "tier3-itransformer-32", "tier3-itransformer-meso"],
    )


def main() -> None:
    args = make_arg_parser(default_h=DEFAULT_H).parse_args()
    run(
        val_days=args.val_days,
        max_days=args.max_days,
        max_steps=args.max_steps,
        upload=args.upload,
        gpus=args.gpus,
        input_size=args.input_size,
        h=args.h,
        step_size=args.step_size,
        output_dir=args.output_dir,
        member_name=args.member_name or MEMBER_NAME,
    )


if __name__ == "__main__":
    main()
