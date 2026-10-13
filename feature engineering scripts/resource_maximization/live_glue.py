"""
Wires resource_maximization's drift-gated GBM adaptation (§13.1) and
MC-dropout uncertainty (§13.3) into inference/live_loop.py's tick cycle —
without modifying live_loop.py itself, matching this repo's established
non-invasive integration pattern (streaming/live_stream_glue.py).

Why this can't just be an on_tick(record) callback like streaming's: §13.1's
GBM adaptation needs the raw lgb.Booster object and a resolved-label window
to refit against, and §13.3's MC-dropout needs the raw torch model to run
stochastic forward passes — none of that is present in the plain summary
dict live_loop.run_forever(on_tick=...) hands its callback (just
ts/prob_down/prob_flat/.../fusion_stage, no member internals). So this
module drives its own copy of live_loop.LiveLoopState/run_tick() directly —
exactly like streaming/live_stream_glue.py's own --once branch does —
reaching into state.members["gbm-family"].booster and
state.members["ssm-challenger"].model after each prediction tick.

Every piece degrades gracefully and independently, matching this repo's
"skip this piece, log it, keep going" pattern used throughout: no
gbm-family checkpoint loaded -> adaptation is skipped that tick; no
training_data_snapshot.parquet -> drift can't be scored, adaptation is
skipped; no ssm-challenger checkpoint -> uncertainty is skipped. None of
this ever blocks or crashes the underlying prediction tick itself.

GBM adaptation needs LABELED data (X, y), which the live tick loop's own
LiveDataBuffer deliberately does not carry (it holds the freshest, still
label-unresolved bars needed for prediction — see live_loop.LiveDataBuffer's
own docstring). So the drift/adaptation window here is loaded independently
via common.data.load_days + filter_valid, capped at
config.LIVE_GLUE_ADAPTATION_LOOKBACK_DAYS days, and checked only once every
config.LIVE_GLUE_DRIFT_CHECK_EVERY_N_TICKS ticks — re-scoring drift and
refitting a booster against nearly the same window on every single tick
would waste cycles for no benefit, since PSI drift is a slow-moving
quantity by construction.

Can be chained with an external on_tick (e.g. streaming's) via the optional
`external_on_tick` passthrough in run_forever() — this module's own
tick-processing runs first, then the external callback receives the same
record, augmented with a "resource_maximization" key holding whatever ran
this tick.

Never run against a real trained checkpoint or real Kaggle-downloaded data
in this repo's own tests — every function is exercised offline against a
synthetic LiveLoopState-like stand-in with mocked members, per this
project's Kaggle-training-excluded-for-now scope.

Usage:
    python -m resource_maximization.live_glue                 # run forever
    python -m resource_maximization.live_glue --once            # single tick
    python -m resource_maximization.live_glue --no-adaptation   # MC-dropout only, skip GBM adaptation
"""
from __future__ import annotations

import argparse
import logging
import time
from typing import Callable

import polars as pl

import config
from common.data import filter_valid, load_days, to_xy
from common.feature_contract import FEATURE_COLUMNS
from inference import live_loop
from resource_maximization.drift_gate import ReplayBuffer, build_adaptation_batch, compute_drift_score, should_adapt
from resource_maximization.mc_dropout_uncertainty import cheap_uncertainty
from resource_maximization.online_adapters import adapt_gbm_member

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("resource_maximization.live_glue")

GBM_SNAPSHOT_PATH = (
    live_loop.loaders.MODELS_ROOT / "gbm_family" / "checkpoints" / "training_data_snapshot.parquet"
)


class ResourceMaximizationState:
    """Wraps one live_loop.LiveLoopState with the extra state this glue
    module needs across ticks: a running tick counter (drift-check cadence),
    the drift reference window (loaded once, not every tick), and a replay
    buffer accumulating resolved-label rows for future adaptation batches
    (§13.1's anti-forgetting mix)."""

    def __init__(self, buffer_bars: int = live_loop.DEFAULT_BUFFER_BARS, enable_trading: bool = True):
        self.live = live_loop.LiveLoopState(buffer_bars=buffer_bars, enable_trading=enable_trading)
        self.tick_count = 0
        self.reference_df: pl.DataFrame | None = None
        self.replay_buffer = ReplayBuffer()
        # The booster as loaded from the last full retrain: every adaptation
        # starts from it (never from a previous adaptation), so repeated
        # drift-gated cycles can't compound trees or drift arbitrarily far.
        self.gbm_anchor = None
        self.last_seen_ts = None


def _load_reference_df() -> pl.DataFrame | None:
    if not GBM_SNAPSHOT_PATH.exists():
        log.info("live_glue: no training_data_snapshot.parquet at %s — drift scoring unavailable this run", GBM_SNAPSHOT_PATH)
        return None
    try:
        return pl.read_parquet(GBM_SNAPSHOT_PATH)
    except Exception:
        log.exception("live_glue: failed to read %s", GBM_SNAPSHOT_PATH)
        return None


def maybe_adapt_gbm(state: ResourceMaximizationState) -> dict | None:
    """Runs at most once every config.LIVE_GLUE_DRIFT_CHECK_EVERY_N_TICKS
    ticks: scores drift between the GBM member's own training snapshot and
    a recent, already label-resolved window, and — only if that member is
    currently loaded — refits it (drift_gate.should_adapt gates whether the
    refit escalates to continued boosting). Returns a small report dict, or
    None if this tick's cadence doesn't call for a check at all."""
    if state.tick_count % config.LIVE_GLUE_DRIFT_CHECK_EVERY_N_TICKS != 0:
        return None

    if state.reference_df is None:
        state.reference_df = _load_reference_df()
    if state.reference_df is None:
        return {"status": "no_reference_snapshot"}

    gbm_member = state.live.members.get("gbm-family")
    if gbm_member is None:
        return {"status": "gbm_member_not_loaded"}

    try:
        recent_df = filter_valid(load_days(max_days=config.LIVE_GLUE_ADAPTATION_LOOKBACK_DAYS))
    except Exception:
        log.exception("live_glue: failed to load recent labeled data for drift scoring — skipping this cycle")
        return {"status": "recent_data_load_failed"}

    if recent_df.height == 0:
        return {"status": "no_recent_labeled_data"}

    # Only rows not seen in an earlier cycle are new evidence; without new
    # resolved labels there is nothing to adapt on (re-fitting the same
    # window every cycle is pure drift-chasing).
    new_rows = recent_df if state.last_seen_ts is None else recent_df.filter(pl.col("ts") > state.last_seen_ts)
    if new_rows.height == 0:
        return {"status": "no_new_labeled_data"}
    state.last_seen_ts = recent_df["ts"].max()
    # Recent window mixed with older replay rows (§13.1 anti-forgetting),
    # sampled before the new rows join the buffer.
    adapt_df = build_adaptation_batch(recent_df, state.replay_buffer)
    state.replay_buffer.add_df(new_rows)

    drift_result = compute_drift_score(state.reference_df, recent_df, FEATURE_COLUMNS)
    report = {"status": "checked", "drift": drift_result, "adapted": False}

    if not should_adapt(drift_result):
        return report

    try:
        # The booster's own trained column list (metrics.json) — a --with-news
        # model has more columns than the default FEATURE_COLUMNS.
        feature_cols = getattr(gbm_member, "feature_cols", FEATURE_COLUMNS)
        missing = [c for c in feature_cols if c not in adapt_df.columns]
        if missing:
            log.info("live_glue: gbm-family needs %d column(s) the recent frame lacks (e.g. %s) — not adapting",
                     len(missing), missing[:3])
            report["status"] = "adapt_skipped_missing_columns"
            return report
        X_new, y_new, _ts = to_xy(adapt_df, feature_cols=feature_cols)
        if state.gbm_anchor is None:
            state.gbm_anchor = gbm_member.booster
        gbm_member.booster = adapt_gbm_member(state.gbm_anchor, X_new, y_new, drift_result)
        report["adapted"] = True
        log.info("live_glue: gbm-family adapted (mean_psi=%.4f)", drift_result["mean_psi"])
    except Exception:
        log.exception("live_glue: adapt_gbm_member raised — leaving gbm-family booster unchanged")
        report["status"] = "adapt_failed"

    return report


def maybe_estimate_uncertainty(state: ResourceMaximizationState) -> dict | None:
    """MC-dropout cheap uncertainty (§13.3) over ssm-challenger's raw torch
    model, run every tick (unlike GBM adaptation, this is cheap enough not
    to need a cadence knob). Returns None if ssm-challenger isn't loaded or
    the buffer doesn't yet have a full lookback window."""
    ssm_member = state.live.members.get("ssm-challenger")
    if ssm_member is None:
        return None

    df = state.live.buffer.df
    X = ssm_member.window(df) if df is not None else None  # same columns + scaling as its training
    if X is None:
        return None
    x_t = ssm_member.torch.from_numpy(X).to(ssm_member.device)

    try:
        mean_proba, std_proba = cheap_uncertainty(ssm_member.model, x_t)
    except Exception:
        log.exception("live_glue: cheap_uncertainty raised for ssm-challenger — skipping this tick")
        return None

    return {
        "mean_proba": mean_proba[0].tolist(), "std_proba": std_proba[0].tolist(),
        "mean_uncertainty": float(std_proba.mean()),
    }


def run_glued_tick(state: ResourceMaximizationState, enable_adaptation: bool = True) -> dict:
    """One full cycle: the underlying prediction tick, plus whatever
    resource-maximization work is due this tick, folded into one record so
    a caller (or an external on_tick) sees everything from a single call."""
    record = live_loop.run_tick(state.live)
    adaptation_report = maybe_adapt_gbm(state) if enable_adaptation else None
    uncertainty_report = maybe_estimate_uncertainty(state)
    state.tick_count += 1
    record["resource_maximization"] = {
        "gbm_adaptation": adaptation_report, "mc_dropout_uncertainty": uncertainty_report,
    }
    return record


def run_forever(
    buffer_bars: int = live_loop.DEFAULT_BUFFER_BARS,
    enable_trading: bool = True,
    max_cycles: int | None = None,
    enable_adaptation: bool = True,
    external_on_tick: Callable[[dict], None] | None = None,
) -> None:
    state = ResourceMaximizationState(buffer_bars=buffer_bars, enable_trading=enable_trading)
    log.info("live_glue starting (adaptation=%s)", enable_adaptation)
    cycles = 0
    while max_cycles is None or cycles < max_cycles:
        tick_start = time.monotonic()
        try:
            record = run_glued_tick(state, enable_adaptation=enable_adaptation)
            if external_on_tick is not None:
                try:
                    external_on_tick(record)
                except Exception:
                    log.exception("external_on_tick callback raised — continuing loop regardless")
        except Exception:
            log.exception("Unhandled error in run_glued_tick — skipping this cycle, loop continues")
        cycles += 1
        elapsed = time.monotonic() - tick_start
        time.sleep(max(0.0, config.LIVE_TICK_INTERVAL_SECONDS - elapsed))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="single glued tick, for smoke-testing wiring")
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--buffer-bars", type=int, default=live_loop.DEFAULT_BUFFER_BARS)
    parser.add_argument("--no-trading", action="store_true")
    parser.add_argument("--no-adaptation", action="store_true", help="skip drift-gated GBM adaptation, MC-dropout uncertainty only")
    args = parser.parse_args()

    if args.once:
        state = ResourceMaximizationState(buffer_bars=args.buffer_bars, enable_trading=not args.no_trading)
        record = run_glued_tick(state, enable_adaptation=not args.no_adaptation)
        log.info("Single glued tick complete: %s", record.get("ts"))
    else:
        run_forever(
            buffer_bars=args.buffer_bars, enable_trading=not args.no_trading,
            max_cycles=args.max_cycles, enable_adaptation=not args.no_adaptation,
        )


if __name__ == "__main__":
    main()

