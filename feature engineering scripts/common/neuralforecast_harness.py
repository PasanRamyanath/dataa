"""
Shared training/eval harness for every neuralforecast-based (github.com/
Nixtla/neuralforecast) ensemble member: models/{tft,patchtst,itransformer,
nhits}/train.py. Each of those is a thin wrapper that picks a model class
and hyperparameters and calls run_member() here — this is where the actual
data-format adaptation, fit/predict loop, evaluation, and checkpointing
lives, so the four members stay mutually consistent by construction instead
of by convention.

neuralforecast requires "long format": one row per (series, timestamp) with
columns unique_id/ds/y plus exogenous columns declared in the model's
hist_exog_list/futr_exog_list/stat_exog_list. We have a single series
(BTCUSDT), so unique_id is constant; `y` is the primary-bar close price
(models forecast the price path itself over FORECAST_PERIODS steps —
matching config.LOOKBACK_PERIODS/FORECAST_PERIODS, confirmed against
train_tft_kaggle.py's own defaults). Exogenous columns are split via
common.feature_contract.FUTURE_KNOWN_FEATURE_COLUMNS /
HISTORICAL_ONLY_FEATURE_COLUMNS so only genuinely-future-knowable calendar
features are ever passed as futr_exog_list — and only to models whose class
declares exogenous support (EXOGENOUS_HIST/EXOGENOUS_FUTR; e.g. PatchTST and
iTransformer are y-only in neuralforecast and would refuse them).

Train/eval is ONE nf.cross_validation(refit=False) call over the contiguous
series that ends at the val window: the model is fit once on everything
before the first cutoff, then forecasts an h-step window from every
config.SEQ_OOF_STEP_BARS-th bar across val, each from the real input_size
bars preceding its cutoff. The OOF signal is keyed at the cutoff (the bar
the forecast was made at) and is the forecast end-of-horizon return vs the
cutoff close — exactly what inference/checkpoint_loaders computes live from
the latest bar. (predict() only forecasts h steps past the END of the frame it is
given, so it cannot score a val window.) The fit region includes the
purge/embargo zone between the tabular train and val splits — harmless for a
price forecaster, whose training targets are closes that all lie before the
first cutoff; nothing in that zone is scored. The frame is the unfiltered
load_days output (never filter_valid, which would punch a 1-hour hole into
every day); if a real day is missing upstream, only the run after the last
gap is used and the truncation is logged.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import polars as pl
import torch

import config
from anti_leakage import Fold
from common.windowing import run_ids
from common.feature_contract import FUTURE_KNOWN_FEATURE_COLUMNS, HISTORICAL_ONLY_FEATURE_COLUMNS
from common.oof import regression_to_direction_proba, save_oof

log = logging.getLogger("common.neuralforecast_harness")

UNIQUE_ID = config.SYMBOL_UM


def resolve_gpu_kwargs(gpus: int | None) -> dict[str, Any]:
    """Resolves accelerator/devices kwargs for the PyTorch-Lightning Trainer
    every neuralforecast model constructs internally from its **trainer_kwargs.

    `gpus=None` auto-detects via torch.cuda.device_count() (0 on this CPU dev
    box, 2 on a Kaggle dual-T4 VM). Deliberately never sets `strategy`: an
    explicit `ddp_spawn` strategy is what breaks NeuralForecast.predict()
    after a multi-GPU fit (confirmed via Nixtla/neuralforecast#1037) — the
    issue's own confirmed workaround is to omit `strategy` entirely and let
    PyTorch Lightning pick its default multi-device strategy, which does not
    exhibit the bug."""
    if gpus is None:
        gpus = torch.cuda.device_count()
    if gpus <= 0:
        return {"accelerator": "cpu", "devices": 1}
    if gpus == 1:
        return {"accelerator": "gpu", "devices": 1}
    return {"accelerator": "gpu", "devices": gpus}


def to_long_format(df: pl.DataFrame, exog_cols: list[str] | None = None):
    """polars -> pandas long-format frame neuralforecast expects. Casts to
    float32/float64 as needed; NeuralForecast internally requires float y and
    numeric exog columns, so booleans/small ints from
    CATEGORICAL_OR_FLAG_FEATURE_COLUMNS are cast to float here too (they're
    still just numeric covariates to these architectures, unlike the GBM
    member which treats them as native categoricals)."""
    if exog_cols is None:
        exog_cols = HISTORICAL_ONLY_FEATURE_COLUMNS + FUTURE_KNOWN_FEATURE_COLUMNS
    out = df.select(
        pl.col("ts").alias("ds"),
        pl.col("close").cast(pl.Float64).fill_nan(0.0).fill_null(0.0).alias("y"),
        *[pl.col(c).cast(pl.Float32).fill_nan(0.0).fill_null(0.0) for c in exog_cols],
    ).with_columns(pl.lit(UNIQUE_ID).alias("unique_id"))
    try:
        return out.to_pandas().copy()
    except Exception:
        import pandas as pd
        return pd.DataFrame(out.to_dict(as_series=False)).copy()


def supported_exog(model_cls) -> tuple[list[str], list[str]]:
    """(hist_exog_list, futr_exog_list) this model class accepts."""
    name = getattr(model_cls, "__name__", "")
    if name in ("PatchTST", "iTransformer"):
        return [], []
    hist = HISTORICAL_ONLY_FEATURE_COLUMNS if getattr(model_cls, "EXOGENOUS_HIST", True) else []
    futr = FUTURE_KNOWN_FEATURE_COLUMNS if getattr(model_cls, "EXOGENOUS_FUTR", True) else []
    return list(hist), list(futr)


def build_model(model_cls, h: int, input_size: int, extra_kwargs: dict[str, Any] | None = None):
    import inspect
    hist, futr = supported_exog(model_cls)
    cls_name = getattr(model_cls, "__name__", "")
    # v4.0 Fix: NHITS has no VSN; feeding 184 sparse exog columns under RobustScaler
    # causes near-zero IQR division ($54,212 MAE explosion). Use standard scaling and
    # prune NHITS historical exog to top non-sparse continuous features.
    scaler_kind = "standard" if cls_name == "NHITS" else "robust"
    if cls_name == "NHITS" and len(hist) > 15:
        core_nhits_exog = [
            c for c in (
                "log_return_1", "log_return_3", "log_return_6", "realized_vol_15",
                "realized_vol_90", "vwap_deviation", "order_flow_imbalance_near",
                "ofi_level_0.2", "ofi_imbalance_decay_slope", "taker_buy_sell_imbalance",
                "basis_pct", "funding_rate", "open_interest_change_pct",
                "toptrader_sum_ls_ratio", "taker_ls_vol_ratio",
            )
            if c in hist
        ]
        hist = core_nhits_exog if core_nhits_exog else hist[:15]
    kwargs = dict(h=h, input_size=input_size, scaler_type=scaler_kind)
    try:
        sig = inspect.signature(model_cls.__init__)
        if "inference_windows_batch_size" in sig.parameters:
            kwargs["inference_windows_batch_size"] = 16
    except Exception:
        pass
    if hist:
        kwargs["hist_exog_list"] = hist
    if futr:
        kwargs["futr_exog_list"] = futr
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    return model_cls(**kwargs)


def directional_accuracy(y_true_path: np.ndarray, y_pred_path: np.ndarray, y_last_known: np.ndarray) -> float:
    """Fraction of forecast windows where the predicted end-of-horizon price
    move has the same sign as the actual move, relative to the last known
    price before the forecast window started — the metric that actually maps
    onto label_triple_barrier's directional target, not raw price MAE."""
    true_sign = np.sign(y_true_path[:, -1] - y_last_known)
    pred_sign = np.sign(y_pred_path[:, -1] - y_last_known)
    return float(np.mean(true_sign == pred_sign))


def eval_series(df: pl.DataFrame, fold: Fold, h: int, input_size: int,
                step: int = config.SEQ_OOF_STEP_BARS) -> tuple[pl.DataFrame, int]:
    """(series, test_size): the contiguous run of `df` that contains the last
    val row, cut to [run start, fold.val_end), plus the number of trailing
    rows cross_validation should score (h plus a multiple of `step`, so
    every cutoff gets a full h-step window)."""
    series = df.slice(0, fold.val_end)
    rid = run_ids(series, config.PRIMARY_INTERVAL_SECONDS)
    run_start = int(np.searchsorted(rid, rid[-1]))
    if run_start > 0:
        log.warning("Gap(s) in the loaded days: using only the last contiguous run (%d of %d rows) — "
                    "%d earlier rows are unused", series.height - run_start, series.height, run_start)
        series = series.slice(run_start)
    val_rows = fold.val_end - max(fold.val_start, run_start)
    max_test = series.height - input_size - h  # leave at least one full training window
    avail = min(val_rows, max_test)
    test_size = h + ((avail - h) // step) * step if avail >= h else 0
    if test_size < h:
        raise ValueError(f"Not enough contiguous data: {series.height} rows in the last run, {val_rows} of them "
                         f"val — need input_size+h={input_size + h} training rows plus at least h={h} val rows")
    return series, test_size


def run_member(
    member_name: str,
    model_cls,
    df: pl.DataFrame,
    fold: Fold,
    h: int = config.FORECAST_PERIODS,
    input_size: int = config.LOOKBACK_PERIODS,
    max_steps: int = 1000,
    extra_model_kwargs: dict[str, Any] | None = None,
    checkpoint_dir: Path | None = None,
    gpus: int | None = None,
) -> dict:
    """`df` is the full unfiltered load_days frame and `fold` the val fold
    (common.cv.purged_walk_forward_splits(...)[0].fold)."""
    import pandas as pd
    from neuralforecast import NeuralForecast
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    gpu_kwargs = resolve_gpu_kwargs(gpus)
    log.info("%s: trainer kwargs %s (torch.cuda.device_count()=%s)",
              member_name, gpu_kwargs, torch.cuda.device_count())
    model = build_model(model_cls, h=h, input_size=input_size,
                         extra_kwargs={"max_steps": max_steps, **gpu_kwargs, **(extra_model_kwargs or {})})
    hist, futr = supported_exog(model_cls)
    used_hist = getattr(model, "hist_exog_list", hist) or []
    used_futr = getattr(model, "futr_exog_list", futr) or []
    nf = NeuralForecast(models=[model], freq=f"{config.PRIMARY_INTERVAL_SECONDS}s")

    step = config.SEQ_OOF_STEP_BARS
    series, test_size = eval_series(df, fold, h, input_size, step)
    long_df = to_long_format(series, list(used_hist) + list(used_futr))
    log.info("%s: fit on %d rows, scoring %d val rows in %d windows of h=%d every %d bars (exog: %d hist, %d futr)",
              member_name, len(long_df) - test_size, test_size, (test_size - h) // step + 1, h, step,
              len(used_hist), len(used_futr))
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        cv = nf.cross_validation(df=long_df, n_windows=None, test_size=test_size, step_size=step,
                                  val_size=0, refit=False)
    except Exception as exc:
        if gpu_kwargs.get("devices", 1) > 1:
            raise RuntimeError(
                f"{member_name}: cross_validation() failed with a {gpu_kwargs['devices']}-GPU trainer. "
                "If this is the known neuralforecast multi-GPU inference issue (Nixtla/neuralforecast#1037), "
                "rerun this member with --gpus 1 (models.train_all does this automatically)."
            ) from exc
        raise

    if "unique_id" not in cv.columns:
        cv = cv.reset_index()
    model_alias = model.__class__.__name__
    pred_col = model_alias if model_alias in cv.columns else [
        c for c in cv.columns if c not in ("unique_id", "ds", "cutoff", "y", "index")][0]

    truth = long_df[["ds", "y"]].rename(columns={"y": "y_true"})
    merged = cv[["ds", "cutoff", pred_col]].merge(truth, on="ds", how="inner").sort_values("ds").reset_index(drop=True)
    if merged.empty:
        raise RuntimeError(f"{member_name}: cross_validation returned no rows matching the series' timestamps")
    merged["cutoff_close"] = merged["cutoff"].map(truth.set_index("ds")["y_true"])

    y_true = merged["y_true"].to_numpy()
    y_pred = merged[pred_col].to_numpy()
    last = merged.groupby("cutoff", sort=True).tail(1)
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "directional_accuracy": directional_accuracy(
            last["y_true"].to_numpy()[:, None], last[pred_col].to_numpy()[:, None], last["cutoff_close"].to_numpy()),
        "n_forecast_points": int(len(merged)),
        "n_windows": int(merged["cutoff"].nunique()),
        "model_alias": model_alias,
    }
    log.info("%s: val MAE=%.4f RMSE=%.4f dir_acc=%.3f over %d points / %d windows", member_name,
              metrics["mae"], metrics["rmse"], metrics["directional_accuracy"], len(merged), metrics["n_windows"])

    # OOF wiring: one row per cutoff, keyed at the cutoff itself (the last
    # bar the forecast could see) — never at a forecast point's ds, whose
    # label looks forward from a bar the forecast was made before. The
    # signal is the forecast end-of-horizon return vs the cutoff close,
    # converted to a soft 3-class distribution (common.oof.
    # regression_to_direction_proba), same as the live predictor.
    cutoff_close = last["cutoff_close"].to_numpy()
    pred_return = (last[pred_col].to_numpy() - cutoff_close) / np.where(cutoff_close != 0, cutoff_close, 1.0)
    direction_proba = regression_to_direction_proba(pred_return)
    save_oof(member_name, pd.to_datetime(last["cutoff"]).to_numpy(), direction_proba)

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        nf.save(path=str(checkpoint_dir), model_index=None, overwrite=True, save_dataset=False)

    return {"metrics": metrics, "nf": nf}
