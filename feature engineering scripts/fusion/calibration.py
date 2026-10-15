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


class TrueDirichletODIRCalibrator:
    """Multiclass Dirichlet Calibrator with Off-Diagonal and Intercept Regularization (ODIR).
    Reference: Kull, Perello-Nieto, et al. (NeurIPS 2019).
    Penalizes only off-diagonal weights (j != k) and intercepts b_k, leaving diagonal
    weights unpenalized to preserve probability sharpness and prevent base-rate collapse.
    """
    def __init__(self, l2_off_diag: float = 1e-2, l2_intercept: float = 1e-3, eps: float = 1e-12):
        self.l2_off_diag = l2_off_diag
        self.l2_intercept = l2_intercept
        self.eps = eps
        self.W = None
        self.b = None
        self.n_classes = None

    def fit(self, proba: np.ndarray, y: np.ndarray) -> "TrueDirichletODIRCalibrator":
        from scipy.optimize import minimize

        N, K = proba.shape
        self.n_classes = K
        ln_p = np.log(np.clip(proba, self.eps, 1.0))
        y_onehot = np.eye(K)[y]

        # Initial parameters: W = Identity, b = zeros
        init_params = np.concatenate([np.eye(K).ravel(), np.zeros(K)])

        def loss_and_grad(params):
            W = params[:K * K].reshape(K, K)
            b = params[K * K:]

            logits = ln_p @ W.T + b
            logits_max = np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            nll = -np.mean(np.sum(y_onehot * np.log(np.clip(softmax, self.eps, 1.0)), axis=1))

            off_diag_mask = ~np.eye(K, dtype=bool)
            reg_off = 0.5 * self.l2_off_diag * np.sum(W[off_diag_mask] ** 2)
            reg_b = 0.5 * self.l2_intercept * np.sum(b ** 2)
            loss = nll + reg_off + reg_b

            diff = (softmax - y_onehot) / N
            grad_W = diff.T @ ln_p
            grad_W[off_diag_mask] += self.l2_off_diag * W[off_diag_mask]

            grad_b = np.sum(diff, axis=0) + self.l2_intercept * b
            return loss, np.concatenate([grad_W.ravel(), grad_b])

        res = minimize(loss_and_grad, init_params, jac=True, method="L-BFGS-B", options={"maxiter": 500})
        self.W = res.x[:K * K].reshape(K, K)
        self.b = res.x[K * K:]
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        ln_p = np.log(np.clip(proba, self.eps, 1.0))
        logits = ln_p @ self.W.T + self.b
        logits -= np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"type": "dirichlet_odir", "W": self.W, "b": self.b, "n_classes": self.n_classes}, f)

    @classmethod
    def load(cls, path: Path) -> "TrueDirichletODIRCalibrator":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls()
        obj.W = state["W"]
        obj.b = state["b"]
        obj.n_classes = state.get("n_classes", 3)
        return obj


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
            pickle.dump({"type": "isotonic", "calibrators": self.calibrators, "n_classes": self.n_classes}, f)

    @classmethod
    def load(cls, path: Path):
        with open(path, "rb") as f:
            state = pickle.load(f)
        if state.get("type") == "dirichlet_odir":
            obj = TrueDirichletODIRCalibrator()
            obj.W = state["W"]
            obj.b = state["b"]
            obj.n_classes = state.get("n_classes", 3)
            return obj
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

    dirichlet = TrueDirichletODIRCalibrator().fit(proba_fit, y_fit)
    proba_eval_dirichlet = dirichlet.transform(proba_eval)
    ece_dirichlet = _expected_calibration_error(proba_eval_dirichlet, y_eval)
    brier_dirichlet = _brier_score(proba_eval_dirichlet, y_eval)

    isotonic = PerClassCalibrator(n_classes=N_CLASSES).fit(proba_fit, y_fit)
    proba_eval_isotonic = isotonic.transform(proba_eval)
    ece_isotonic = _expected_calibration_error(proba_eval_isotonic, y_eval)
    brier_isotonic = _brier_score(proba_eval_isotonic, y_eval)

    log.info(
        "Calibration comparison: Dirichlet ODIR (ECE=%.4f, Brier=%.4f) vs Isotonic (ECE=%.4f, Brier=%.4f)",
        ece_dirichlet, brier_dirichlet, ece_isotonic, brier_isotonic,
    )

    if brier_dirichlet <= brier_isotonic:
        calibrator = dirichlet
        proba_eval_calibrated = proba_eval_dirichlet
        calibrator_type = "dirichlet_odir"
    else:
        calibrator = isotonic
        proba_eval_calibrated = proba_eval_isotonic
        calibrator_type = "isotonic"

    metrics = {
        "fit_rows": calib_fit.height,
        "eval_rows": calib_eval.height,
        "calibrator_type": calibrator_type,
        "brier_before": _brier_score(proba_eval, y_eval),
        "brier_after": _brier_score(proba_eval_calibrated, y_eval),
        "ece_before": _expected_calibration_error(proba_eval, y_eval),
        "ece_after": _expected_calibration_error(proba_eval_calibrated, y_eval),
        "dirichlet_ece": ece_dirichlet,
        "dirichlet_brier": brier_dirichlet,
        "isotonic_ece": ece_isotonic,
        "isotonic_brier": brier_isotonic,
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
