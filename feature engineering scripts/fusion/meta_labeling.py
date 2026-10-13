"""
Meta-labeling filter (Lopez de Prado-style bet sizing) — the second fusion
stage the plan calls for alongside the "regime-gated stacking meta-learner":
where fusion.stacking_meta_learner decides WHICH direction to bet, this
module decides WHETHER to act on that bet at all, and how large, by
predicting label_meta_act (the pipeline's own precomputed "was the primary
directional call actually worth acting on" label) from the primary model's
own predicted probabilities + confidence + regime.

Critically, this is fit only on rows the stacking meta-learner itself never
trained on: fusion.stacking_meta_learner.run() carves its OOF frame into
(meta_train, meta_val) by time; this module reloads that exact split, takes
ONLY meta_val (genuinely out-of-sample from the primary model's point of
view), and further splits THAT chronologically (with its own purge gap) into
a meta-labeling train/val. This mirrors Lopez de Prado's requirement that the
meta-labeling model never be fit on the primary model's in-sample
predictions, which would let it learn the primary's overfitting instead of
its genuine skill.

bet_size() turns "probability the primary call is worth acting on" into a
position-size multiplier in [0, 1] by combining act-probability with the
primary model's own confidence (top-class prob minus second-best) — a
low-confidence primary call is sized down even if the filter says "act".

Usage:
    python -m fusion.meta_labeling --upload
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
log = logging.getLogger("fusion.meta_labeling")

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints" / "meta_labeling_filter"
MEMBER_NAME = "fusion-meta-labeling-filter"


def build_bet_sizing_dataset(learner: StackingMetaLearner, meta_df: pl.DataFrame, max_days: int | None) -> pl.DataFrame:
    """`meta_df` must already be rows the primary stacking learner never
    trained on. Derives primary_prob_{down,flat,up}, primary_confidence
    (top minus second-best class prob), and primary_direction from the
    primary model's predictions, then joins the true act/no-act label."""
    proba = learner.predict_proba(meta_df)
    sorted_proba = np.sort(proba, axis=1)
    confidence = sorted_proba[:, -1] - sorted_proba[:, -2]
    primary_direction = proba.argmax(axis=1)

    labels = (
        filter_valid(load_days(max_days=max_days))
        .select(["ts", "label_meta_act"])
        .rename({"label_meta_act": "act"})
        .with_columns(pl.col("act").cast(pl.Int64))
    )

    regime_cols = [c for c in meta_df.columns if c.startswith("regime_")]
    out = meta_df.select(["ts", *regime_cols]).with_columns([
        pl.Series("primary_prob_down", proba[:, 0]),
        pl.Series("primary_prob_flat", proba[:, 1]),
        pl.Series("primary_prob_up", proba[:, 2]),
        pl.Series("primary_confidence", confidence),
        pl.Series("primary_direction", primary_direction.astype(np.int64)),
    ])
    joined = out.join(labels, on="ts", how="inner")
    if joined.height == 0:
        raise ValueError("build_bet_sizing_dataset: no overlap between primary predictions and label_meta_act")
    return joined


class MetaLabelFilter:
    def __init__(self, feature_cols: list[str], model=None):
        self.feature_cols = feature_cols
        if model is not None:
            self.model = model
        else:
            from sklearn.linear_model import LogisticRegression

            self.model = LogisticRegression(max_iter=2000, C=1.0)

    def fit(self, df: pl.DataFrame, y: np.ndarray) -> "MetaLabelFilter":
        X = df.select(self.feature_cols).to_numpy().astype(np.float64)
        self.model.fit(X, y)
        return self

    def act_probability(self, df: pl.DataFrame) -> np.ndarray:
        X = df.select(self.feature_cols).to_numpy().astype(np.float64)
        raw = self.model.predict_proba(X)
        classes = getattr(self.model, "classes_", np.array([0, 1]))
        full = np.zeros((raw.shape[0], 2), dtype=np.float64)
        for i, c in enumerate(classes):
            full[:, int(c)] = raw[:, i]
        return full[:, 1]

    def bet_size(self, df: pl.DataFrame) -> np.ndarray:
        """Position-size multiplier in [0, 1]: act-probability scaled down
        by the primary model's own confidence, so a filter that says "act"
        on a low-confidence primary call still sizes small."""
        act_p = self.act_probability(df)
        confidence = df["primary_confidence"].to_numpy()
        return np.clip(act_p * confidence, 0.0, 1.0)

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "feature_cols": self.feature_cols}, f)

    @classmethod
    def load(cls, path: Path) -> "MetaLabelFilter":
        with open(path, "rb") as f:
            state = pickle.load(f)
        return cls(feature_cols=state["feature_cols"], model=state["model"])


def run(val_frac: float, purge_bars: int, meta_val_frac: float, max_days: int | None, upload: bool) -> dict:
    stacking_path = STACKING_CHECKPOINT_DIR / "stacking_meta_learner.pkl"
    if not stacking_path.exists():
        raise FileNotFoundError(
            f"No stacking meta-learner checkpoint at {stacking_path} — run "
            "`python -m fusion.stacking_meta_learner` first; meta-labeling filters its output."
        )
    primary = StackingMetaLearner.load(stacking_path)

    oof_dict = load_all_oof()
    regime_df = load_regime_states()
    meta_df = build_meta_features(oof_dict, regime_df)
    labeled = attach_labels(meta_df, max_days=max_days)

    # Same chronological carve stacking_meta_learner.run() used — meta_val
    # here is exactly the rows the primary model never trained on.
    _, primary_val = _time_split(labeled, val_frac, purge_bars)
    primary_val = primary_val.drop("y")

    bet_df = build_bet_sizing_dataset(primary, primary_val, max_days=max_days)
    # _time_split only splits by row position — no dependency on a "y"
    # column name — so it applies directly to bet_df (labeled "act").
    meta_train, meta_val = _time_split(bet_df, meta_val_frac, purge_bars)

    feature_cols = [c for c in bet_df.columns if c not in ("ts", "act")]
    y_train = meta_train["act"].to_numpy()
    y_val = meta_val["act"].to_numpy()

    filt = MetaLabelFilter(feature_cols=feature_cols)
    filt.fit(meta_train, y_train)
    act_p_val = filt.act_probability(meta_val)
    pred_val = (act_p_val >= 0.5).astype(int)
    bet_sizes = filt.bet_size(meta_val)

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    metrics = {
        "feature_cols": feature_cols,
        "train_rows": meta_train.height,
        "val_rows": meta_val.height,
        "positive_rate_train": float(y_train.mean()) if len(y_train) else None,
        "positive_rate_val": float(y_val.mean()) if len(y_val) else None,
        "accuracy": float(accuracy_score(y_val, pred_val)),
        "precision": float(precision_score(y_val, pred_val, zero_division=0)),
        "recall": float(recall_score(y_val, pred_val, zero_division=0)),
        "f1": float(f1_score(y_val, pred_val, zero_division=0)),
        "mean_bet_size": float(bet_sizes.mean()) if len(bet_sizes) else None,
        "mean_bet_size_when_acted": float(bet_sizes[pred_val == 1].mean()) if (pred_val == 1).any() else None,
    }
    log.info(
        "Meta-labeling filter: accuracy=%.4f precision=%.4f recall=%.4f mean_bet_size=%.4f",
        metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["mean_bet_size"] or 0.0,
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = CHECKPOINT_DIR / "meta_labeling_filter.pkl"
    filt.save(model_path)

    training_data_path = CHECKPOINT_DIR / "training_data_snapshot.parquet"
    meta_train.write_parquet(training_data_path, compression="snappy")
    (CHECKPOINT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log.info("Saved meta-labeling filter checkpoint to %s", model_path)

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
    parser.add_argument("--meta-val-frac", type=float, default=0.3)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    run(args.val_frac, args.purge_bars, args.meta_val_frac, args.max_days, args.upload)


if __name__ == "__main__":
    main()
