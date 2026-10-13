"""
Drift-magnitude estimation and adaptation gating (§13.1, plan-marked
OPTIONAL). Between full retraining cycles, this module answers one
question for online_adapters.py: "has the live feature distribution moved
enough from what a member was trained on to justify an adaptation pass?" —
never "adapt on every tick," which the plan explicitly calls out as the
drift-chasing failure mode documented in 2024/25 incremental-learning
benchmarks (a Numerai-style incremental framework still showed large
drawdowns through genuine regime breaks even with continuous adaptation).

Method: per-feature Population Stability Index (PSI) between a reference
window (the member's own training-data snapshot, e.g.
models/gbm_family/checkpoints/training_data_snapshot.parquet) and a recent
live window, aggregated by taking the mean across features. PSI is a
standard, cheap, distribution-agnostic drift metric (credit-risk/ML-monitoring
convention: <0.1 no significant shift, 0.1-0.2 moderate, >0.2 significant) —
chosen over a KS-test or KL-divergence here specifically because it's bounded,
symmetric enough for monitoring use, and doesn't require a parametric
distribution assumption, which matters across FEATURE_COLUMNS' very
different scales/shapes (ratios, log-returns, counts, flags).

Honesty note carried over from the plan: this technique is well-established
for monitoring, but the specific "adapt when PSI crosses threshold" recipe
for THIS ensemble has not been validated against a simpler periodic-retrain-
only baseline — that validation is §10's job once real training exists, not
this module's.

ReplayBuffer implements §13.1's anti-forgetting safeguard: a small buffer of
older, diverse examples online_adapters.py mixes into every adaptation batch
so an adapted member never drifts arbitrarily far from its last full-retrain
anchor.

Usage:
    from resource_maximization.drift_gate import compute_drift_score, should_adapt, ReplayBuffer
    score = compute_drift_score(reference_df, recent_df, feature_cols)
    if should_adapt(score):
        ...

    python -m resource_maximization.drift_gate --demo   # synthetic data only, no real checkpoints touched
"""
from __future__ import annotations

import logging

import numpy as np
import polars as pl

import config

log = logging.getLogger("resource_maximization.drift_gate")


def _psi_one_feature(reference: np.ndarray, recent: np.ndarray, bins: int) -> float:
    """PSI for a single feature: bin the reference distribution into
    `bins` equal-frequency buckets, then compare bucket population shares
    between reference and recent. Returns 0.0 (no evidence of drift) when
    either sample is too small/degenerate to bin meaningfully — a module
    that can't measure drift must say so via a neutral score, never crash
    the caller."""
    reference = reference[np.isfinite(reference)]
    recent = recent[np.isfinite(recent)]
    if len(reference) < config.DRIFT_PSI_MIN_SAMPLES or len(recent) < config.DRIFT_PSI_MIN_SAMPLES:
        return 0.0

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 3:  # degenerate (near-constant) feature — PSI is not meaningful
        return 0.0

    # Extend the outer edges to +/-inf so recent values outside the reference's
    # historical min/max still land in the extreme bins instead of being
    # silently dropped by np.histogram (which only counts values inside the
    # given bin range) — a live window moving into genuinely unprecedented
    # territory is exactly the case PSI must flag, not undercount.
    open_edges = edges.copy()
    open_edges[0] = -np.inf
    open_edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=open_edges)
    rec_counts, _ = np.histogram(recent, bins=open_edges)

    ref_frac = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-6, None)
    rec_frac = np.clip(rec_counts / max(rec_counts.sum(), 1), 1e-6, None)

    return float(np.sum((rec_frac - ref_frac) * np.log(rec_frac / ref_frac)))


def compute_drift_score(
    reference_df: pl.DataFrame,
    recent_df: pl.DataFrame,
    feature_cols: list[str],
    bins: int = config.DRIFT_PSI_BINS,
) -> dict:
    """Returns {"mean_psi": float, "per_feature": {col: psi}, "n_features_scored": int}.
    Features missing from either frame, all-null, or too small a sample are
    silently skipped (not scored as 0 drift, excluded from the mean) rather
    than crashing or biasing the aggregate toward "no drift"."""
    per_feature: dict[str, float] = {}
    for col in feature_cols:
        if col not in reference_df.columns or col not in recent_df.columns:
            continue
        ref_vals = reference_df[col].cast(pl.Float64, strict=False).drop_nulls().to_numpy()
        rec_vals = recent_df[col].cast(pl.Float64, strict=False).drop_nulls().to_numpy()
        if len(ref_vals) == 0 or len(rec_vals) == 0:
            continue
        per_feature[col] = _psi_one_feature(ref_vals, rec_vals, bins)

    mean_psi = float(np.mean(list(per_feature.values()))) if per_feature else 0.0
    return {"mean_psi": mean_psi, "per_feature": per_feature, "n_features_scored": len(per_feature)}


def should_adapt(drift_result: dict, threshold: float = config.DRIFT_PSI_THRESHOLD) -> bool:
    return drift_result["mean_psi"] >= threshold


class ReplayBuffer:
    """Fixed-capacity store of older, diverse rows (§13.1's anti-forgetting
    safeguard). `add()` reservoir-samples so the buffer stays a roughly
    uniform sample of everything ever added, not just the most recent
    additions — the opposite bias of a plain ring buffer, which is what
    makes it useful alongside recent-data updates rather than redundant
    with them."""

    def __init__(self, capacity: int = config.REPLAY_BUFFER_SIZE, seed: int = 0):
        self.capacity = capacity
        self._rows: list[dict] = []
        self._n_seen = 0
        self._rng = np.random.default_rng(seed)

    def add(self, rows: list[dict]) -> None:
        for row in rows:
            self._n_seen += 1
            if len(self._rows) < self.capacity:
                self._rows.append(row)
            else:
                j = int(self._rng.integers(0, self._n_seen))
                if j < self.capacity:
                    self._rows[j] = row

    def add_df(self, df: pl.DataFrame) -> None:
        self.add(df.to_dicts())

    def sample(self, n: int) -> list[dict]:
        if not self._rows:
            return []
        n = min(n, len(self._rows))
        idx = self._rng.choice(len(self._rows), size=n, replace=False)
        return [self._rows[i] for i in idx]

    def __len__(self) -> int:
        return len(self._rows)


def build_adaptation_batch(
    recent_df: pl.DataFrame,
    replay_buffer: ReplayBuffer,
    recent_fraction: float = config.REPLAY_BUFFER_RECENT_FRACTION,
) -> pl.DataFrame:
    """Mixes `recent_df` with a replay-buffer sample per §13.1's anti-
    forgetting recipe, returning one combined frame ready for
    online_adapters.py to adapt on. If the buffer is empty (e.g. first-ever
    adaptation this session), returns recent_df unchanged rather than
    failing — there's nothing to anchor against yet."""
    if len(replay_buffer) == 0 or recent_df.height == 0:
        return recent_df
    n_recent = max(1, int(recent_df.height * recent_fraction))
    n_replay = max(1, recent_df.height - n_recent)
    recent_part = recent_df.tail(n_recent)
    replay_rows = replay_buffer.sample(n_replay)
    if not replay_rows:
        return recent_part
    replay_part = pl.DataFrame(replay_rows)
    common_cols = [c for c in recent_part.columns if c in replay_part.columns]
    return pl.concat([recent_part.select(common_cols), replay_part.select(common_cols)], how="vertical_relaxed")


def _demo() -> None:
    rng = np.random.default_rng(42)
    reference_df = pl.DataFrame({"f1": rng.normal(0, 1, 2000), "f2": rng.normal(5, 2, 2000)})
    stable_recent = pl.DataFrame({"f1": rng.normal(0, 1, 500), "f2": rng.normal(5, 2, 500)})
    drifted_recent = pl.DataFrame({"f1": rng.normal(3, 1, 500), "f2": rng.normal(5, 2, 500)})

    stable_score = compute_drift_score(reference_df, stable_recent, ["f1", "f2"])
    drifted_score = compute_drift_score(reference_df, drifted_recent, ["f1", "f2"])
    print("stable window drift score:", stable_score, "-> adapt:", should_adapt(stable_score))
    print("drifted window drift score:", drifted_score, "-> adapt:", should_adapt(drifted_score))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    if args.demo:
        _demo()
    else:
        parser.print_help()
