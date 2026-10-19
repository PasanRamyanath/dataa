"""
Regime-gated stacking meta-learner — the layer that turns 9 independently
trained ensemble members into the one combined signal the plan's Appendix A
calls for ("regime-gated stacking meta-learner + meta-labeling filter").

Each member's train.py writes its held-out val-window predictions to
fusion/oof/{member}.parquet via common.oof.save_oof (direction members) or
common.oof.save_regime_states (hmm_regime, which gates rather than votes).
This module:

  1. build_meta_features() backward-as-of-joins every member's (ts,
     prob_down, prob_flat, prob_up) frame on ts — prefixing columns with the
     member name so they never collide — and adds a one-hot-encoded regime
     state as extra gating features the meta-learner can learn to weight
     members differently by regime.
  2. attach_labels() joins the true label_triple_barrier for those exact ts
     values back in from the feature-engineered dataset, so the meta-learner
     is trained supervised, not just averaging members blindly.
  3. StackingMetaLearner wraps a small classifier (logistic regression by
     default — interpretable, low-variance, appropriate given how few rows
     typically survive the OOF intersection) over the joined member
     probabilities + regime one-hot, with fit/predict_proba/save/load.

Only ever trained on OOF (held-out) predictions — never on any member's own
training-fold predictions — so the meta-learner cannot learn to exploit a
member's overfitting to its own train split.

Usage:
    python -m fusion.stacking_meta_learner --upload
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
from common.data import filter_valid, load_days
from common.oof import load_all_oof, load_regime_states

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fusion.stacking_meta_learner")

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints" / "stacking_meta_learner"
MEMBER_NAME = "fusion-stacking-meta-learner"
PROB_COLS = ("prob_down", "prob_flat", "prob_up")


def build_meta_features(
    oof_dict: dict[str, pl.DataFrame],
    regime_df: pl.DataFrame | None = None,
    n_states: int | None = None,
    max_staleness_bars: int = config.OOF_MAX_STALENESS_BARS,
) -> pl.DataFrame:
    """One row per ts of the densest member's OOF grid, carrying every
    member's prob_down/prob_flat/prob_up (prefixed with its name). Sequence
    members score val on a strided grid, so each other member contributes
    its latest prediction made AT OR BEFORE that ts (backward as-of join —
    causal, like a live tick that reuses a slower member's last vote) and
    rows where some member's latest prediction is older than
    `max_staleness_bars` are dropped rather than zero-filled, which would
    look like a confident "flat" vote it never made. Regime one-hot columns
    are sized by `n_states` (else the regime file's n_states column, else
    max observed + 1) so training and a single live row agree."""
    if not oof_dict:
        raise ValueError("build_meta_features: no OOF predictions found under fusion/oof/ — run member train.py scripts first")

    members = sorted(oof_dict.keys())
    base = max(members, key=lambda m: oof_dict[m].height)
    tolerance = f"{max_staleness_bars * config.PRIMARY_INTERVAL_SECONDS}s"
    meta = oof_dict[base].sort("ts").rename({c: f"{base}__{c}" for c in PROB_COLS})
    for member in members:
        if member == base:
            continue
        renamed = oof_dict[member].sort("ts").rename({c: f"{member}__{c}" for c in PROB_COLS})
        meta = meta.join_asof(renamed, on="ts", strategy="backward", tolerance=tolerance)
    meta = meta.drop_nulls()

    if regime_df is not None and regime_df.height > 0:
        if n_states is None:
            n_states = (int(regime_df["n_states"][0]) if "n_states" in regime_df.columns
                        else int(regime_df["regime_state"].max()) + 1)
        regime_onehot = regime_df.select(["ts", "regime_state"]).with_columns([
            (pl.col("regime_state") == s).cast(pl.Int8).alias(f"regime_{s}") for s in range(n_states)
        ]).drop("regime_state")
        meta = meta.join_asof(regime_onehot.sort("ts"), on="ts", strategy="backward", tolerance=tolerance)
        regime_cols = [f"regime_{s}" for s in range(n_states)]
        meta = meta.with_columns([pl.col(c).fill_null(0).cast(pl.Int8) for c in regime_cols])

    return meta.sort("ts")


def attach_labels(meta_df: pl.DataFrame, max_days: int | None = None) -> pl.DataFrame:
    """Joins the true label_triple_barrier (remapped {-1,0,1}->{0,1,2}) back
    onto the meta-feature frame by ts, from the same feature-engineered
    dataset every member trained against."""
    labels = (
        filter_valid(load_days(max_days=max_days))
        .select(["ts", "label_triple_barrier"])
        .with_columns((pl.col("label_triple_barrier").cast(pl.Int64) + 1).alias("y"))
        .select(["ts", "y"])
    )
    joined = meta_df.join(labels, on="ts", how="inner")
    if joined.height == 0:
        raise ValueError("attach_labels: no overlap between meta-feature ts values and the labeled dataset")
    return joined


class StackingMetaLearner:
    def __init__(self, feature_cols: list[str], model=None):
        self.feature_cols = feature_cols
        if model is not None:
            self.model = model
        else:
            from sklearn.linear_model import LogisticRegression

            self.model = LogisticRegression(max_iter=2000, C=1.0)

    def fit(self, meta_df: pl.DataFrame, y: np.ndarray) -> "StackingMetaLearner":
        X = meta_df.select(self.feature_cols).to_numpy().astype(np.float64)
        self.model.fit(X, y)
        return self

    def predict_proba(self, meta_df: pl.DataFrame) -> np.ndarray:
        """Always returns a full (n, 3) [down, flat, up] array, even if
        y_train happened not to contain every class — sklearn classifiers
        only emit columns for classes seen in training, which would
        otherwise silently misalign column meaning downstream (e.g. in
        log_loss or any consumer assuming column i == class i)."""
        # Impute missing member columns with neutral prior [0.33, 0.34, 0.33] if some members are absent
        cols_to_add = {}
        for c in self.feature_cols:
            if c not in meta_df.columns:
                if c.endswith("__prob_down") or c.endswith("__prob_up"):
                    cols_to_add[c] = pl.lit(0.33, dtype=pl.Float64)
                elif c.endswith("__prob_flat"):
                    cols_to_add[c] = pl.lit(0.34, dtype=pl.Float64)
                else:
                    cols_to_add[c] = pl.lit(0, dtype=pl.Int8)
        if cols_to_add:
            meta_df = meta_df.with_columns(**cols_to_add)

        X = meta_df.select(self.feature_cols).to_numpy().astype(np.float64)
        raw = self.model.predict_proba(X)
        classes = getattr(self.model, "classes_", np.arange(raw.shape[1]))
        full = np.zeros((raw.shape[0], 3), dtype=np.float64)
        for i, c in enumerate(classes):
            full[:, int(c)] = raw[:, i]
        return full

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "feature_cols": self.feature_cols}, f)

    @classmethod
    def load(cls, path: Path) -> "StackingMetaLearner":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(feature_cols=state["feature_cols"], model=state["model"])
        return obj


def _time_split(meta_df: pl.DataFrame, val_frac: float, purge_bars: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Simple chronological split (not full purged walk-forward — the OOF
    store is already only held-out val-window data from each member, so this
    just carves a final slice of it for meta-learner validation, with a
    purge gap so no near-duplicate ts sits on both sides of the boundary)."""
    n = meta_df.height
    n_val = max(int(n * val_frac), 1)
    val_start = n - n_val
    train_end = max(val_start - purge_bars, 0)
    train = meta_df.slice(0, train_end)
    val = meta_df.slice(val_start, n_val)
    if train.height == 0:
        raise ValueError("_time_split: not enough OOF rows to carve a non-empty train split — collect more OOF data")
    return train, val


def run(val_frac: float, purge_bars: int, max_days: int | None, upload: bool) -> dict:
    oof_dict = load_all_oof()
    regime_df = load_regime_states()
    meta_df = build_meta_features(oof_dict, regime_df)
    labeled = attach_labels(meta_df, max_days=max_days)

    feature_cols = [c for c in labeled.columns if c not in ("ts", "y")]
    train_df, val_df = _time_split(labeled, val_frac, purge_bars)

    y_train = train_df["y"].to_numpy()
    y_val = val_df["y"].to_numpy()

    learner = StackingMetaLearner(feature_cols=feature_cols)
    learner.fit(train_df, y_train)
    proba_val = learner.predict_proba(val_df)
    pred_val = proba_val.argmax(axis=1)

    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss

    metrics = {
        "members": sorted(oof_dict.keys()),
        "has_regime_gating": regime_df is not None,
        "feature_cols": feature_cols,
        "train_rows": train_df.height,
        "val_rows": val_df.height,
        "accuracy": float(accuracy_score(y_val, pred_val)),
        "balanced_accuracy": float(balanced_accuracy_score(y_val, pred_val)),
        "f1_macro": float(f1_score(y_val, pred_val, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y_val, proba_val, labels=[0, 1, 2])),
    }
    log.info(
        "Stacking meta-learner: accuracy=%.4f balanced_accuracy=%.4f f1_macro=%.4f (members=%s)",
        metrics["accuracy"], metrics["balanced_accuracy"], metrics["f1_macro"], metrics["members"],
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = CHECKPOINT_DIR / "stacking_meta_learner.pkl"
    learner.save(model_path)

    training_data_path = CHECKPOINT_DIR / "training_data_snapshot.parquet"
    train_df.write_parquet(training_data_path, compression="snappy")
    (CHECKPOINT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log.info("Saved stacking meta-learner checkpoint to %s", model_path)

    if upload:
        from common.hf_model_upload import upload_best_checkpoint, upload_training_data_snapshot

        repo_id = upload_best_checkpoint(MEMBER_NAME, [model_path, CHECKPOINT_DIR / "metrics.json"], metrics=metrics)
        data_repo_id = upload_training_data_snapshot(MEMBER_NAME, [training_data_path])
        log.info("Uploaded checkpoint to %s and training data to %s", repo_id, data_repo_id)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--purge-bars", type=int, default=config.PURGE_BARS)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    run(args.val_frac, args.purge_bars, args.max_days, args.upload)


if __name__ == "__main__":
    main()
