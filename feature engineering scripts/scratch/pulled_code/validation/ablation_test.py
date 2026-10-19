"""
Generic "with vs without a feature branch" ablation comparator (§9.3
sentiment ablation, §13.2 multi-asset ablation) — lives in validation/
rather than resource_maximization/ since it's generic infrastructure used
by any optional branch, not something specific to the resource-maximization
modules themselves.

Shares validation/regime_holdout_eval.py's exact evidence bar (>= 2 of 3
non-overlapping slices, via the same build_validation_slices()) so a
proposed variant is held to the identical standard a member itself is
under §10.3's kill-criteria — the only difference is *what* two things are
being compared: a member vs. the naive majority-class baseline there, a
variant-with-the-branch vs. a variant-without-it here.
config.ABLATION_RELATIVE_LIFT_THRESHOLD defaults to the same 0.01 ("~1%
relative") as regime_holdout_eval.DEFAULT_RELATIVE_THRESHOLD, though the two
are independent knobs.

This module does not know or care what the ablation IS. It only requires
both sides be OOF-shaped per common/oof.py's (ts, prob_down, prob_flat,
prob_up) convention and registered under whatever name the caller used with
common.oof.save_oof() — e.g. save_oof("sentiment_branch_with", ...) and
save_oof("sentiment_branch_without", ...), then
`python -m validation.ablation_test --variant sentiment_branch_with
--baseline sentiment_branch_without`. No new storage format, no new file
layout: producing an ablation-testable branch is exactly what every member's
train.py already does once per side.

Usage:
    python -m validation.ablation_test --variant multi_asset_btc --baseline ssm_challenger
    python -m validation.ablation_test --variant sentiment_branch_with --baseline sentiment_branch_without
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import polars as pl

import config
from common.data import filter_valid, load_days
from common.oof import load_all_oof
from validation.naive_baselines import _classification_metrics
from validation.regime_holdout_eval import REQUIRED_PASS_SLICES, _to_naive_utc, build_validation_slices

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("validation.ablation_test")

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints" / "ablation_test"


def _direction_predictions(oof_df: pl.DataFrame, labels_df: pl.DataFrame):
    """Joins one side's OOF probabilities against true labels and returns
    (ts, y_true, y_pred), or None if there's no overlap at all — mirrors
    regime_holdout_eval.evaluate_member's join/argmax logic exactly so the
    two report the same numbers for the same OOF file."""
    joined = _to_naive_utc(oof_df).join(labels_df, on="ts", how="inner")
    if joined.height == 0:
        return None
    y_true = joined["label_triple_barrier"].to_numpy().astype(np.int64) + 1
    proba = joined.select(["prob_down", "prob_flat", "prob_up"]).to_numpy()
    y_pred = proba.argmax(axis=1)
    ts = joined["ts"].to_numpy()
    return ts, y_true, y_pred


def compare_variant_to_baseline(
    variant_df: pl.DataFrame,
    baseline_df: pl.DataFrame,
    labels_df: pl.DataFrame,
    relative_threshold: float = config.ABLATION_RELATIVE_LIFT_THRESHOLD,
) -> dict:
    """Compares balanced-accuracy of `variant_df` against `baseline_df`
    across >= 2 of 3 non-overlapping slices, evaluated ONLY on timestamps
    both sides actually cover — a variant that happens to cover a different
    (easier) time range than its baseline must not get credit for it."""
    variant_parsed = _direction_predictions(variant_df, labels_df)
    baseline_parsed = _direction_predictions(baseline_df, labels_df)
    if variant_parsed is None or baseline_parsed is None:
        return {"status": "no_overlap_with_labels"}

    v_ts, v_y_true, v_y_pred = variant_parsed
    b_ts, b_y_true, b_y_pred = baseline_parsed

    common_ts = np.intersect1d(v_ts, b_ts)
    if len(common_ts) == 0:
        return {"status": "no_shared_timestamps"}

    ts_min = common_ts.min().astype("datetime64[us]").item()
    ts_max = common_ts.max().astype("datetime64[us]").item()
    slices = build_validation_slices(ts_min, ts_max)

    v_index = {t: i for i, t in enumerate(v_ts)}
    b_index = {t: i for i, t in enumerate(b_ts)}

    slice_results = []
    passed = 0
    for sl in slices:
        start = np.datetime64(sl["start"])
        end = np.datetime64(sl["end"])
        mask = (common_ts >= start) & (common_ts < end)
        slice_ts = common_ts[mask]
        if len(slice_ts) == 0:
            slice_results.append({**sl, "status": "no_rows_in_slice"})
            continue

        v_idx = np.array([v_index[t] for t in slice_ts])
        b_idx = np.array([b_index[t] for t in slice_ts])
        variant_metrics = _classification_metrics(v_y_true[v_idx], v_y_pred[v_idx])
        baseline_metrics = _classification_metrics(b_y_true[b_idx], b_y_pred[b_idx])

        variant_acc = variant_metrics["balanced_accuracy"]
        baseline_acc = baseline_metrics["balanced_accuracy"]
        required = baseline_acc * (1.0 + relative_threshold)
        beats_baseline = variant_acc >= required if baseline_acc > 0 else variant_acc > baseline_acc
        if beats_baseline:
            passed += 1

        slice_results.append({
            **sl, "status": "evaluated", "n_rows": int(len(slice_ts)),
            "variant_balanced_accuracy": variant_acc,
            "baseline_balanced_accuracy": baseline_acc,
            "required_balanced_accuracy": required,
            "beats_baseline": bool(beats_baseline),
            "variant_metrics": variant_metrics,
            "baseline_metrics": baseline_metrics,
        })

    evaluated = [s for s in slice_results if s["status"] == "evaluated"]
    decision = "adopt_variant" if passed >= REQUIRED_PASS_SLICES else "keep_baseline"
    return {
        "status": "evaluated",
        "relative_threshold": relative_threshold,
        "n_slices_evaluated": len(evaluated),
        "n_slices_total": len(slices),
        "n_slices_passed": passed,
        "required_pass_slices": REQUIRED_PASS_SLICES,
        "decision": decision,
        "slices": slice_results,
    }


def run(
    variant_name: str,
    baseline_name: str,
    max_days: int | None = None,
    relative_threshold: float = config.ABLATION_RELATIVE_LIFT_THRESHOLD,
) -> dict:
    oof = load_all_oof()
    missing = [n for n in (variant_name, baseline_name) if n not in oof]
    if missing:
        report = {
            "status": "missing_oof", "missing": missing,
            "message": f"No OOF file for {missing} yet under fusion/oof/ — save_oof() must be called for "
                       f"both the variant and baseline branches before an ablation comparison can run.",
        }
        log.warning(report["message"])
        return report

    df = filter_valid(load_days(max_days=max_days))
    labels_df = _to_naive_utc(df.select(["ts", "label_triple_barrier"]))

    result = compare_variant_to_baseline(oof[variant_name], oof[baseline_name], labels_df, relative_threshold)
    result["variant"] = variant_name
    result["baseline"] = baseline_name

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CHECKPOINT_DIR / f"{variant_name}_vs_{baseline_name}.json"
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    if result["status"] == "evaluated":
        log.info("%s vs %s: %d/%d slices passed -> %s", variant_name, baseline_name,
                  result["n_slices_passed"], result["n_slices_total"], result["decision"].upper())
    else:
        log.warning("%s vs %s: %s", variant_name, baseline_name, result["status"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, help="OOF name (fusion/oof/<name>.parquet) of the branch being proposed")
    parser.add_argument("--baseline", required=True, help="OOF name of what the variant must beat")
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--relative-threshold", type=float, default=config.ABLATION_RELATIVE_LIFT_THRESHOLD)
    args = parser.parse_args()
    run(args.variant, args.baseline, args.max_days, args.relative_threshold)


if __name__ == "__main__":
    main()
