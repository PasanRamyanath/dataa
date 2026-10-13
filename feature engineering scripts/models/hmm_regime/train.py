"""
HMM regime classifier ensemble member: a Gaussian-emission Hidden Markov
Model over a small set of volatility/trend/order-flow features, producing a
discrete regime state per bar. Not a directional predictor itself — its
output (the regime posterior / most-likely state) is meant to be consumed as
a gating input by fusion/stacking_meta_learner.py, matching the plan's
"regime-gated stacking meta-learner" design.

Deliberately uses only a handful of already-causal, already-trailing
features (never label columns) so the regime label at bar t depends only on
information available at or before t — same anti-leakage guarantee as every
other member, just via a much smaller and more interpretable feature set
than the full 216-column tabular contract.

Usage:
    python -m models.hmm_regime.train --n-states 3 --upload
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
from common.cv import purged_walk_forward_splits
from common.data import filter_valid, load_days

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("models.hmm_regime.train")

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"

REGIME_FEATURES = [
    "log_return_1",
    "realized_vol_90",
    "realized_vol_360",
    "momentum_180",
    "taker_buy_sell_imbalance",
    "funding_rate",
]


def _matrix(df: pl.DataFrame) -> np.ndarray:
    X = df.select(REGIME_FEATURES).with_columns([pl.col(c).cast(pl.Float64).fill_null(0.0) for c in REGIME_FEATURES])
    return X.to_numpy()


def train_hmm(X_train: np.ndarray, n_states: int, seed: int = 42):
    from hmmlearn.hmm import GaussianHMM

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    X_norm = (X_train - mean) / std

    model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=200, random_state=seed)
    model.fit(X_norm)
    return model, mean, std


def evaluate(model, mean: np.ndarray, std: np.ndarray, X_val: np.ndarray) -> dict:
    X_norm = (X_val - mean) / std
    log_likelihood = model.score(X_norm)
    states = model.predict(X_norm)
    return {
        "val_log_likelihood": float(log_likelihood),
        "val_log_likelihood_per_bar": float(log_likelihood / max(len(X_val), 1)),
        "state_occupancy": {str(s): int(c) for s, c in zip(*np.unique(states, return_counts=True))},
    }


def run(n_states: int, n_folds: int, val_days: int, max_days: int | None, upload: bool) -> dict:
    df = filter_valid(load_days(max_days=max_days))
    val_size = val_days * config.BARS_PER_DAY
    splits = purged_walk_forward_splits(df, n_folds=n_folds, val_size=val_size)

    cv_report = {"folds": []}
    for split in splits:
        X_train = _matrix(split.train)
        X_val = _matrix(split.val)
        model, mean, std = train_hmm(X_train, n_states)
        metrics = evaluate(model, mean, std, X_val)
        cv_report["folds"].append({"fold_index": split.fold_index, **metrics})
        log.info("Fold %d: val_log_likelihood_per_bar=%.4f", split.fold_index, metrics["val_log_likelihood_per_bar"])

    final_split = splits[-1]
    X_train_final = _matrix(final_split.train)
    final_model, mean, std = train_hmm(X_train_final, n_states)
    final_metrics = evaluate(final_model, mean, std, _matrix(final_split.val))

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = CHECKPOINT_DIR / "hmm_regime_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": final_model, "mean": mean, "std": std, "features": REGIME_FEATURES, "n_states": n_states}, f)

    training_data_path = CHECKPOINT_DIR / "training_data_snapshot.parquet"
    final_split.train.write_parquet(training_data_path, compression="snappy")

    report = {"cv": cv_report, "final_fold": final_metrics, "features": REGIME_FEATURES, "n_states": n_states}
    (CHECKPOINT_DIR / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("Saved HMM checkpoint to %s", model_path)

    from common.oof import save_regime_states

    X_val_final = _matrix(final_split.val)
    val_states = final_model.predict((X_val_final - mean) / std)
    save_regime_states(final_split.val["ts"].to_numpy(), val_states)

    if upload:
        from common.hf_model_upload import upload_best_checkpoint, upload_training_data_snapshot

        repo_id = upload_best_checkpoint("hmm-regime", [model_path, CHECKPOINT_DIR / "metrics.json"], metrics=report)
        data_repo_id = upload_training_data_snapshot("hmm-regime", [training_data_path])
        log.info("Uploaded checkpoint to %s and training data to %s", repo_id, data_repo_id)

    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-states", type=int, default=3)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--val-days", type=int, default=config.VAL_DAYS)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    run(args.n_states, args.n_folds, args.val_days, args.max_days, args.upload)


if __name__ == "__main__":
    main()
