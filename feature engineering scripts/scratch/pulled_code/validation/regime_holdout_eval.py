"""
Regime-holdout kill-criteria automation per plan §10.3: "A model member that
does not beat the naive baseline by the reporting threshold [~1% relative,
per §10.1], across at least two of three non-overlapping validation slices,
is excluded from the live ensemble (logged, re-evaluated next cycle)."

This is the piece that turns raw per-member metrics into an actual go/no-go
decision. It intentionally does NOT parse each member's checkpoints/
metrics.json — those have genuinely different schemas per member (gbm-family
nests cv.folds[i].direction.accuracy, hmm-regime has
cv.folds[i].val_log_likelihood_per_bar, sentiment-branch has a top-level
accuracy, etc. — confirmed by direct inspection of every models/*/train.py).
Instead it works off the one schema every direction-predicting member agrees
on: common/oof.py's (ts, prob_down, prob_flat, prob_up) store, joined back
against the true label_triple_barrier for those same timestamps. hmm-regime
is a gating input, not a directional vote (see common/oof.py docstring), so
it is excluded from kill-criteria evaluation here, same as it's excluded
from fusion/stacking_meta_learner.py's member list.

Validation slices (plan §10.1 wants >= 3 non-overlapping slices, one of
which is the 2021-2023 regime-diversity holdout):
  - This pipeline's actual downloaded/backfilled data has not been confirmed
    to include 2021-2023 (only ~2026 sample data was confirmed present as of
    this writing). Rather than silently fabricate that slice or crash, this
    script checks the loaded data's real ts range: if any rows fall in
    2021-01-01..2023-12-31 it carves that out as the "regime_diversity"
    slice and splits the remainder into 2 more contiguous time slices; if
    none do, it logs an explicit, honest gap notice (per the plan's own
    §12.4 "honest gap reporting" philosophy) and falls back to 3 contiguous
    tertiles of whatever range IS available, clearly labeled as a
    substitute — not a hidden stand-in — for the mandated regime-diversity
    holdout.

Naive baseline per slice: majority-class accuracy computed from every row
strictly BEFORE that slice's start (the same "majority of the train fold"
definition validation/naive_baselines.py uses), falling back to the whole
pre-OOF dataset's majority class if a slice starts at the very beginning of
available data.

A member with no OOF file yet (not trained this cycle, or never trained) is
reported as "no_oof_data" rather than failed — this script is meant to be
run repeatedly as members finish training (mirrors models/train_all.py's
resilience: a missing artifact is an honest "not yet", not a crash).

Usage:
    python -m validation.regime_holdout_eval
    python -m validation.regime_holdout_eval --relative-threshold 0.01
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

import config
from common.data import filter_valid, load_days
from common.oof import load_all_oof
from validation.naive_baselines import _classification_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("validation.regime_holdout_eval")

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints" / "regime_holdout_eval"

# All datetimes in this module are naive-but-UTC (no tzinfo attached), not
# tz-aware: common/oof.py's save_oof() stores `ts` as a plain numpy array
# (common.data.to_xy does `df["ts"].to_numpy()`), which silently drops
# polars' tz-aware Datetime metadata on the numpy round-trip — every real
# member's OOF parquet is therefore tz-naive even though the feature
# pipeline's own ts column (common.data.load_days) is tz-aware UTC. Rather
# than fight that mismatch member-by-member, this script normalizes
# everything to naive-UTC once (see _to_naive_utc below) and stays naive
# throughout.
REGIME_DIVERSITY_START = datetime(2021, 1, 1)
REGIME_DIVERSITY_END = datetime(2024, 1, 1)  # exclusive upper bound (through 2023-12-31)

DEFAULT_RELATIVE_THRESHOLD = 0.01  # "~1% relative" per plan §10.1
REQUIRED_PASS_SLICES = 2           # "at least two of three" per plan §10.3


def _to_naive_utc(df: pl.DataFrame, col: str = "ts") -> pl.DataFrame:
    dtype = df.schema[col]
    if isinstance(dtype, pl.Datetime) and dtype.time_zone is not None:
        return df.with_columns(pl.col(col).dt.replace_time_zone(None))
    return df


def _slice_label(name: str, start: datetime, end: datetime) -> dict:
    return {"name": name, "start": start.isoformat(), "end": end.isoformat()}


def build_validation_slices(ts_min: datetime, ts_max: datetime) -> list[dict]:
    """Returns >= 3 non-overlapping [start, end) time slices covering the
    available OOF ts range, carving out the 2021-2023 regime-diversity
    holdout as its own slice when the data actually covers it."""
    has_regime_diversity = ts_min < REGIME_DIVERSITY_END and ts_max >= REGIME_DIVERSITY_START

    if has_regime_diversity:
        slices = [_slice_label("regime_diversity_holdout_2021_2023", REGIME_DIVERSITY_START, REGIME_DIVERSITY_END)]
        # Remainder = whatever of [ts_min, ts_max) falls outside the regime-diversity window.
        remainder_bounds = []
        if ts_min < REGIME_DIVERSITY_START:
            remainder_bounds.append((ts_min, REGIME_DIVERSITY_START))
        if ts_max >= REGIME_DIVERSITY_END:
            remainder_bounds.append((REGIME_DIVERSITY_END, ts_max))
        if not remainder_bounds:
            # entire dataset IS the regime-diversity window; split it into 3 tertiles instead
            log.warning("Entire available OOF range falls inside 2021-2023 — treating it as 3 contiguous "
                        "tertiles rather than 1 slice + remainder.")
            return _tertile_slices(ts_min, ts_max, label_prefix="regime_diversity_tertile")
        for i, (start, end) in enumerate(remainder_bounds):
            slices.append(_slice_label(f"post_holdout_segment_{i}", start, end))
        # If only one remainder segment, split it in two so we still have >= 3 slices total.
        if len(slices) < 3:
            last = slices.pop()
            start = datetime.fromisoformat(last["start"])
            end = datetime.fromisoformat(last["end"])
            mid = start + (end - start) / 2
            slices.append(_slice_label(last["name"] + "_a", start, mid))
            slices.append(_slice_label(last["name"] + "_b", mid, end))
        return slices

    log.warning(
        "GAP: loaded feature data does not cover the plan's mandated 2021-2023 regime-diversity holdout "
        "window (§10.1/§10.3) — available OOF ts range is [%s, %s]. Falling back to 3 contiguous tertiles "
        "of the available range as a SUBSTITUTE, not a replacement, for that requirement. Backfill "
        "2021-2023 data before treating a pass/fail decision from this run as satisfying §10.3 in full.",
        ts_min.isoformat(), ts_max.isoformat(),
    )
    return _tertile_slices(ts_min, ts_max, label_prefix="available_range_tertile")


def _tertile_slices(ts_min: datetime, ts_max: datetime, label_prefix: str) -> list[dict]:
    total = (ts_max - ts_min) / 3
    bounds = [ts_min + total * i for i in range(4)]
    return [_slice_label(f"{label_prefix}_{i}", bounds[i], bounds[i + 1]) for i in range(3)]


def _majority_baseline_accuracy(y_train: np.ndarray, y_slice: np.ndarray) -> dict:
    if len(y_train) == 0:
        y_train = y_slice  # only true for a slice starting at the very beginning of available data
    majority = int(np.bincount(y_train.astype(np.int64), minlength=3).argmax())
    pred = np.full_like(y_slice, majority)
    return _classification_metrics(y_slice, pred)


def evaluate_member(
    member_name: str,
    oof_df: pl.DataFrame,
    labels_df: pl.DataFrame,
    slices: list[dict],
    relative_threshold: float,
) -> dict:
    joined = _to_naive_utc(oof_df).join(labels_df, on="ts", how="inner")
    if joined.height == 0:
        return {"status": "no_overlap_with_labels", "member": member_name}

    y_true_all = (joined["label_triple_barrier"].to_numpy().astype(np.int64) + 1)
    proba = joined.select(["prob_down", "prob_flat", "prob_up"]).to_numpy()
    y_pred_all = proba.argmax(axis=1)
    ts_all = joined["ts"].to_numpy()

    slice_results = []
    passed = 0
    for sl in slices:
        start = np.datetime64(sl["start"])
        end = np.datetime64(sl["end"])
        mask = (ts_all >= start) & (ts_all < end)
        n = int(mask.sum())
        if n == 0:
            slice_results.append({**sl, "status": "no_oof_rows_in_slice"})
            continue

        y_true_slice = y_true_all[mask]
        y_pred_slice = y_pred_all[mask]
        member_metrics = _classification_metrics(y_true_slice, y_pred_slice)

        train_mask = ts_all < start
        baseline_metrics = _majority_baseline_accuracy(y_true_all[train_mask], y_true_slice)

        baseline_acc = baseline_metrics["balanced_accuracy"]
        member_acc = member_metrics["balanced_accuracy"]
        required = baseline_acc * (1.0 + relative_threshold)
        beats_baseline = member_acc >= required if baseline_acc > 0 else member_acc > baseline_acc
        if beats_baseline:
            passed += 1

        slice_results.append({
            **sl, "status": "evaluated", "n_rows": n,
            "member_balanced_accuracy": member_acc,
            "naive_baseline_balanced_accuracy": baseline_acc,
            "required_balanced_accuracy": required,
            "beats_baseline": bool(beats_baseline),
            "member_metrics": member_metrics,
            "naive_baseline_metrics": baseline_metrics,
        })

    evaluated_slices = [s for s in slice_results if s["status"] == "evaluated"]
    decision = "keep" if passed >= REQUIRED_PASS_SLICES else "exclude"
    return {
        "status": "evaluated",
        "member": member_name,
        "n_slices_evaluated": len(evaluated_slices),
        "n_slices_total": len(slices),
        "n_slices_passed": passed,
        "required_pass_slices": REQUIRED_PASS_SLICES,
        "decision": decision,
        "slices": slice_results,
    }


def run(max_days: int | None, relative_threshold: float) -> dict:
    df = filter_valid(load_days(max_days=max_days))
    labels_df = _to_naive_utc(df.select(["ts", "label_triple_barrier"]))

    oof = load_all_oof()
    if not oof:
        report = {
            "status": "no_oof_data",
            "message": "No member has written fusion/oof/*.parquet yet — run models/*/train.py first "
                       "(this is expected before any Kaggle training has run; see common/oof.py).",
        }
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        (CHECKPOINT_DIR / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        log.warning(report["message"])
        return report

    all_ts = np.concatenate([d["ts"].to_numpy() for d in oof.values()])
    ts_min = all_ts.min().astype("datetime64[us]").item()
    ts_max = all_ts.max().astype("datetime64[us]").item()
    slices = build_validation_slices(ts_min, ts_max)

    member_reports = {}
    for member_name, oof_df in sorted(oof.items()):
        result = evaluate_member(member_name, oof_df, labels_df, slices, relative_threshold)
        member_reports[member_name] = result
        if result["status"] == "evaluated":
            log.info("%s: %d/%d slices passed -> %s", member_name, result["n_slices_passed"],
                      result["n_slices_total"], result["decision"].upper())
        else:
            log.warning("%s: %s", member_name, result["status"])

    kept = [m for m, r in member_reports.items() if r.get("decision") == "keep"]
    excluded = [m for m, r in member_reports.items() if r.get("decision") == "exclude"]

    report = {
        "status": "evaluated",
        "relative_threshold": relative_threshold,
        "required_pass_slices": REQUIRED_PASS_SLICES,
        "oof_ts_range": {"min": ts_min.isoformat(), "max": ts_max.isoformat()},
        "slices": slices,
        "members": member_reports,
        "kept_members": kept,
        "excluded_members": excluded,
        "note": "hmm-regime is intentionally excluded from this evaluation — it's a regime-gating input, "
                "not a directional vote (see common/oof.py / fusion/stacking_meta_learner.py).",
    }

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    (CHECKPOINT_DIR / "metrics.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    log.info("Wrote regime-holdout kill-criteria report to %s — keep=%s exclude=%s", CHECKPOINT_DIR, kept, excluded)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--relative-threshold", type=float, default=DEFAULT_RELATIVE_THRESHOLD,
                         help="member must beat naive baseline balanced-accuracy by this relative margin (plan §10.1: ~1%%)")
    args = parser.parse_args()
    run(args.max_days, args.relative_threshold)


if __name__ == "__main__":
    main()
