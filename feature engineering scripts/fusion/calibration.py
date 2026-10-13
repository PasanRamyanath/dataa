"""
Per-class isotonic calibration for the fusion layer's final combined output.

A stacking meta-learner's raw predict_proba (fusion.stacking_meta_learner)
is not guaranteed to be well-calibrated — e.g. "70% up" bucket predictions
may not actually resolve up 70% of the time — which matters directly for
meta-labeling's bet sizing (fusion.meta_labeling multiplies act-probability
by primary confidence, so a miscalibrated confidence silently mis-sizes
every bet). This module fits one isotonic regressor per class (one-vs-rest,
sklearn.isotonic.IsotonicRegression), then renormalizes the three calibrated
scores back to a valid probability simplex.

Like fusion.meta_labeling, this is fit only on rows the stacking
meta-learner never trained on: it reloads the exact (train, val) carve
stacking_meta_learner.run() used, takes val (genuinely out-of-sample from
the primary model's point of view), and further splits THAT chronologically
(its own purge gap) into a calibration-fit / calibration-eval split — so
calibration quality is measured out-of-sample too, not on the same rows it
was fit on.

Usage:
    python -m fusion.calibration --upload
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import polars as pl

import config
from common.oof import load_all_oof, load_regime_states
from fusion.stacking_meta_learner import (
    CHECKPOINT_DIR as STACKING_CHECKPOINT_DIR,
)
from fusion.stacking_meta_learner import (
    StackingMetaLearner,
    _time_split,
    attach_labels,
    build_meta_features,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fusion.calibration")

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints" / "calibration"
MEMBER_NAME = "fusion-calibration"
N_CLASSES = 3


class PerClassCalibrator:
    def __init__(self, n_classes: int = N_CLASSES):
        self.n_classes = n_classes
        self.calibrators: list | None = None

    def fit(self, proba: np.ndarray, y: np.ndarray) -> "PerClassCalibrator":
        from sklearn.isotonic import IsotonicRegression

        self.calibrators = []
        for k in range(self.n_classes):
            ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            ir.fit(proba[:, k], (y == k).astype(np.float64))
            self.calibrators.append(ir)
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        if self.calibrators is None:
            raise RuntimeError("PerClassCalibrator.transform called before fit")
        calibrated = np.stack([self.calibrators[k].predict(proba[:, k]) for k in range(self.n_classes)], axis=1)
        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return calibrated / row_sums

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"calibrators": self.calibrators, "n_classes": self.n_classes}, f)

    @classmethod
    def load(cls, path: Path) -> "PerClassCalibrator":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(n_classes=state["n_classes"])
        obj.calibrators = state["calibrators"]
        return obj


def _brier_score(proba: np.ndarray, y: np.ndarray, n_classes: int = N_CLASSES) -> float:
    onehot = np.eye(n_classes)[y]
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def _expected_calibration_error(proba: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Standard multiclass ECE: bins the predicted-class confidence
    (max-probability), and within each bin compares mean confidence to
    empirical accuracy — a well-calibrated model has |gap| near 0 in every
    bin that has enough samples to be meaningful."""
    confidences = proba.max(axis=1)
    predictions = proba.argmax(axis=1)
    correct = (predictions == y).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if not mask.any():
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def run(val_frac: float, purge_bars: int, calib_val_frac: float, max_days: int | None, upload: bool) -> dict:
    stacking_path = STACKING_CHECKPOINT_DIR / "stacking_meta_learner.pkl"
    if not stacking_path.exists():
        raise FileNotFoundError(
            f"No stacking meta-learner checkpoint at {stacking_path} — run "
            "`python -m fusion.stacking_meta_learner` first; calibration wraps its output."
        )
    primary = StackingMetaLearner.load(stacking_path)

    oof_dict = load_all_oof()
    regime_df = load_regime_states()
    meta_df = build_meta_features(oof_dict, regime_df)
    labeled = attach_labels(meta_df, max_days=max_days)

    _, primary_val = _time_split(labeled, val_frac, purge_bars)
    calib_fit, calib_eval = _time_split(primary_val, calib_val_frac, purge_bars)

    proba_fit = primary.predict_proba(calib_fit)
    y_fit = calib_fit["y"].to_numpy()
    proba_eval = primary.predict_proba(calib_eval)
    y_eval = calib_eval["y"].to_numpy()

    calibrator = PerClassCalibrator(n_classes=N_CLASSES).fit(proba_fit, y_fit)
    proba_eval_calibrated = calibrator.transform(proba_eval)

    metrics = {
        "fit_rows": calib_fit.height,
        "eval_rows": calib_eval.height,
        "brier_before": _brier_score(proba_eval, y_eval),
        "brier_after": _brier_score(proba_eval_calibrated, y_eval),
        "ece_before": _expected_calibration_error(proba_eval, y_eval),
        "ece_after": _expected_calibration_error(proba_eval_calibrated, y_eval),
    }
    log.info(
        "Calibration: brier %.4f -> %.4f, ECE %.4f -> %.4f",
        metrics["brier_before"], metrics["brier_after"], metrics["ece_before"], metrics["ece_after"],
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = CHECKPOINT_DIR / "per_class_calibrator.pkl"
    calibrator.save(model_path)

    training_data_path = CHECKPOINT_DIR / "training_data_snapshot.parquet"
    calib_fit.write_parquet(training_data_path, compression="snappy")
    (CHECKPOINT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log.info("Saved calibrator checkpoint to %s", model_path)

    if upload:
        from common.hf_model_upload import upload_best_checkpoint, upload_training_data_snapshot

        repo_id = upload_best_checkpoint(MEMBER_NAME, [model_path, CHECKPOINT_DIR / "metrics.json"], metrics=metrics)
        data_repo_id = upload_training_data_snapshot(MEMBER_NAME, [training_data_path])
        log.info("Uploaded checkpoint to %s and training data to %s", repo_id, data_repo_id)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-frac", type=float, default=0.2, help="must match stacking_meta_learner's --val-frac")
    parser.add_argument("--purge-bars", type=int, default=config.PURGE_BARS)
    parser.add_argument("--calib-val-frac", type=float, default=0.3)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    run(args.val_frac, args.purge_bars, args.calib_val_frac, args.max_days, args.upload)


if __name__ == "__main__":
    main()
