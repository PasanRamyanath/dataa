"""
Live inference loop per plan §12.1 — polls whatever feature-engineered bars
the existing pipeline (build_dataset.py, writing under config.OUTPUT_ROOT)
has produced so far, runs a forward pass through every currently-available
ensemble member, fuses their votes through the trained fusion stage, and
feeds the result to inference/paper_trading_engine.py.

Scope boundary (deliberate, matching this repo's existing pattern of
documenting where a gap-closing script's responsibility stops — see
models/sentiment_branch/train.py's SENTIMENT_PROXY_FEATURES docstring and
config.py's news-ingestion section): this loop consumes the feature
pipeline's OUTPUT, it does not itself connect to a live exchange feed or run
the feature-engineering pipeline — that pipeline already exists elsewhere in
this repo and is a live-exchange-network-touching concern explicitly out of
scope for this gap. "~1Hz live-tick loop" here means: check for new rows in
the feature pipeline's latest output file(s) once a second, and produce a
fresh prediction+trade decision exactly when a new bar actually appears
(never fabricating one when the bar hasn't advanced).

Every member's forward pass and the fusion stage itself are wrapped so a
missing checkpoint or a raised exception degrades that one piece — logged,
skipped for this tick — rather than crashing the loop (see
inference/checkpoint_loaders.py's module docstring for why). Before ANY
member is trained, this loop still runs end-to-end: zero members vote, fused
output is all-None, and no paper trade is placed, logged clearly each tick
rather than silently idling.

Streaming (§14, a separate still-unbuilt gap) hooks in via an optional
`on_tick` callback passed to run_forever()/main() — this module has no
Cloudflare/webhook dependency itself; a later streaming/webhook_dispatcher.py
can pass its own sink function without live_loop.py needing to change.

Usage:
    python -m inference.live_loop                  # run forever at ~1Hz
    python -m inference.live_loop --once             # single tick and exit (testing)
    python -m inference.live_loop --max-cycles 10     # stop after N ticks (testing)
    python -m inference.live_loop --no-trading        # inference only, skip paper-trading engine
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from typing import Callable

import numpy as np
import polars as pl

import config
from common.feature_contract import conform_to_contract
from inference import checkpoint_loaders as loaders
from inference.paper_trading_engine import PaperTradingEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("inference.live_loop")

# Trailing-row buffer size: must cover the largest per-member context window
# (ssm-challenger needs LOOKBACK_PERIODS, kronos-finetune needs up to
# MAX_CONTEXT, neuralforecast members need LOOKBACK_PERIODS as input_size)
# plus a small margin.
DEFAULT_BUFFER_BARS = max(config.LOOKBACK_PERIODS, loaders.MAX_CONTEXT) + 32


class LiveDataBuffer:
    """Tails config.OUTPUT_ROOT's feature parquet files. Deliberately does
    NOT call common.data.filter_valid — filter_valid drops rows whose
    forward-looking labels can't resolve yet (labeling.py's label_is_valid),
    which for a live tail is exactly the most recent, still-open bars this
    loop needs to score. Labels are irrelevant here anyway; only features
    are consumed."""

    def __init__(self, buffer_bars: int = DEFAULT_BUFFER_BARS, symbol: str = config.SYMBOL_UM):
        self.buffer_bars = buffer_bars
        self.symbol = symbol
        self.df: pl.DataFrame | None = None
        self._file_mtimes: dict[str, float] = {}

    def refresh(self) -> bool:
        files = sorted(config.OUTPUT_ROOT.glob(f"features_{self.symbol}_*.parquet"))
        if not files:
            return False
        recent_files = files[-2:]  # today + yesterday covers any buffer_bars <= 1 day of 10s bars comfortably
        changed = False
        for f in recent_files:
            mtime = f.stat().st_mtime
            if self._file_mtimes.get(str(f)) != mtime:
                self._file_mtimes[str(f)] = mtime
                changed = True
        if not changed and self.df is not None:
            return False

        frames = [conform_to_contract(pl.read_parquet(f)) for f in recent_files]
        combined = pl.concat(frames, how="vertical_relaxed").sort("ts")
        self.df = combined.tail(self.buffer_bars)
        return True

    def latest_ts(self):
        if self.df is None or self.df.height == 0:
            return None
        return self.df.tail(1)["ts"][0]


def _fuse(
    member_predictions: dict[str, np.ndarray],
    regime_state: int | None,
    hmm,
    stacking,
    calibrator,
    meta_filter,
) -> dict:
    """Builds the single-row fusion input from whichever members voted this
    tick and runs stacking -> calibration -> meta-labeling for whichever of
    those three fusion-stage checkpoints are actually loaded. Falls back to
    a plain mean vote when the stacking checkpoint isn't available yet (it
    won't be until fusion/stacking_meta_learner.py has been run), or if the
    loaded stacking learner's feature_cols don't match today's available
    member set (e.g. it was fit on 8 members but only 3 are loaded now)."""
    if not member_predictions:
        return {
            "prob_down": None, "prob_flat": None, "prob_up": None, "primary_direction": None,
            "confidence": None, "act_probability": None, "bet_size": None, "fusion_stage": "no_members",
        }

    ts0 = [datetime(2000, 1, 1, tzinfo=timezone.utc)]  # arbitrary shared join key for this single fused row only
    fusion_stage = "mean_vote_no_stacking_checkpoint"
    proba = np.mean(list(member_predictions.values()), axis=0)

    if stacking is not None:
        from fusion.stacking_meta_learner import build_meta_features

        oof_dict = {
            name: pl.DataFrame({"ts": ts0, "prob_down": [float(p[0])], "prob_flat": [float(p[1])], "prob_up": [float(p[2])]})
            for name, p in member_predictions.items()
        }
        regime_df = None
        if regime_state is not None and hmm is not None:
            regime_df = pl.DataFrame({"ts": ts0, "regime_state": [regime_state]})
        try:
            meta_row = build_meta_features(oof_dict, regime_df, n_states=hmm.n_states if hmm is not None else None)
            proba = stacking.predict_proba(meta_row)[0]
            fusion_stage = "stacking"
        except Exception:
            log.exception("Stacking meta-learner predict_proba failed (likely member-set mismatch) — falling back to mean vote")
            proba = np.mean(list(member_predictions.values()), axis=0)
            fusion_stage = "mean_vote_fallback"

    # The meta-labeling filter was trained on the UNcalibrated stacking
    # output (fusion.meta_labeling.build_bet_sizing_dataset), so its inputs
    # and the direction it judges must come from that same raw output.
    raw_proba = np.asarray(proba, dtype=np.float64)

    if calibrator is not None and fusion_stage == "stacking":
        try:
            proba = calibrator.transform(proba[None, :])[0]
            fusion_stage = "stacking+calibrated"
        except Exception:
            log.exception("Calibrator transform failed — using uncalibrated stacking output")

    sorted_p = np.sort(proba)
    confidence = float(sorted_p[-1] - sorted_p[-2])
    # Trade the direction the meta filter actually evaluated (raw argmax);
    # per-class calibration + renormalization can occasionally move argmax.
    primary_direction = int(np.argmax(raw_proba))
    raw_sorted = np.sort(raw_proba)
    raw_confidence = float(raw_sorted[-1] - raw_sorted[-2])

    act_probability = None
    bet_size = None
    if meta_filter is not None and fusion_stage.startswith("stacking"):
        try:
            regime_cols = {}
            if regime_state is not None and hmm is not None:
                regime_cols = {f"regime_{s}": [1 if s == regime_state else 0] for s in range(hmm.n_states)}
            row = pl.DataFrame({
                "ts": ts0, **regime_cols,
                "primary_prob_down": [float(raw_proba[0])], "primary_prob_flat": [float(raw_proba[1])],
                "primary_prob_up": [float(raw_proba[2])], "primary_confidence": [raw_confidence],
                "primary_direction": [primary_direction],
            })
            act_probability = float(meta_filter.act_probability(row)[0])
            bet_size = float(meta_filter.bet_size(row)[0])
        except Exception:
            log.exception("Meta-labeling filter failed (likely feature_cols mismatch) — no act/bet_size this tick")

    return {
        "prob_down": float(proba[0]), "prob_flat": float(proba[1]), "prob_up": float(proba[2]),
        "primary_direction": primary_direction, "confidence": confidence,
        "act_probability": act_probability, "bet_size": bet_size, "fusion_stage": fusion_stage,
    }


class LiveLoopState:
    def __init__(self, buffer_bars: int = DEFAULT_BUFFER_BARS, enable_trading: bool = True):
        self.buffer = LiveDataBuffer(buffer_bars=buffer_bars)
        self.members = loaders.load_all_members()
        self.hmm = loaders.load_hmm_regime()
        self.stacking = loaders.load_stacking_meta_learner()
        self.calibrator = loaders.load_calibrator()
        self.meta_filter = loaders.load_meta_labeling_filter()
        self.trading_engine = PaperTradingEngine() if enable_trading else None
        self.last_processed_ts = None


def _append_tick_log(record: dict) -> None:
    config.LIVE_TICK_LOG_ROOT.mkdir(parents=True, exist_ok=True)
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = config.LIVE_TICK_LOG_ROOT / f"{day}.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def run_tick(state: LiveLoopState) -> dict:
    t0 = time.monotonic()
    advanced = state.buffer.refresh()
    df = state.buffer.df
    if df is None or df.height == 0:
        return {"status": "no_data"}

    latest_ts = state.buffer.latest_ts()
    if not advanced or (state.last_processed_ts is not None and latest_ts == state.last_processed_ts):
        return {"status": "no_new_bar", "ts": str(latest_ts)}
    state.last_processed_ts = latest_ts

    member_predictions: dict[str, np.ndarray] = {}
    member_latencies: dict[str, float] = {}
    for name, predictor in state.members.items():
        m0 = time.monotonic()
        try:
            proba = predictor.predict(df)
        except Exception:
            log.exception("%s: predict() raised — skipping this member for this tick", name)
            proba = None
        member_latencies[name] = round(time.monotonic() - m0, 4)
        if proba is not None:
            member_predictions[name] = proba

    regime_state = None
    if state.hmm is not None:
        try:
            regime_state = state.hmm.predict_regime(df)
        except Exception:
            log.exception("hmm-regime: predict_regime() raised")

    fused = _fuse(member_predictions, regime_state, state.hmm, state.stacking, state.calibrator, state.meta_filter)

    total_latency = time.monotonic() - t0
    if total_latency > config.LIVE_MAX_TICK_LATENCY_SECONDS:
        log.warning("Tick latency %.2fs exceeds LIVE_MAX_TICK_LATENCY_SECONDS=%.1fs (ts=%s)", total_latency, config.LIVE_MAX_TICK_LATENCY_SECONDS, latest_ts)

    tick_record = {
        "ts": str(latest_ts), "status": "scored",
        "n_members_voted": len(member_predictions), "members_voted": sorted(member_predictions),
        "regime_state": regime_state, "latency_seconds": round(total_latency, 4),
        "member_latencies": member_latencies, **fused,
    }

    if state.trading_engine is not None and fused.get("primary_direction") is not None:
        close_price = float(df.tail(1)["close"][0])
        ts_dt = latest_ts if isinstance(latest_ts, datetime) else datetime.now(timezone.utc)
        trade_result = state.trading_engine.on_tick(ts=ts_dt, close_price=close_price, fused=fused, regime_state=regime_state, hmm=state.hmm)
        tick_record["trading"] = {"equity": trade_result["equity"], "position_fraction": trade_result["position_fraction"], "trade": trade_result["trade"]}

    _append_tick_log(tick_record)
    return tick_record


def run_forever(
    buffer_bars: int = DEFAULT_BUFFER_BARS,
    enable_trading: bool = True,
    max_cycles: int | None = None,
    on_tick: Callable[[dict], None] | None = None,
) -> None:
    state = LiveLoopState(buffer_bars=buffer_bars, enable_trading=enable_trading)
    log.info(
        "Live loop starting: %d/%d direction members loaded, hmm-regime=%s, stacking=%s, calibrator=%s, meta_filter=%s, trading=%s",
        len(state.members), len(loaders.DIRECTION_MEMBER_LOADERS), state.hmm is not None,
        state.stacking is not None, state.calibrator is not None, state.meta_filter is not None, enable_trading,
    )
    cycles = 0
    while max_cycles is None or cycles < max_cycles:
        tick_start = time.monotonic()
        try:
            record = run_tick(state)
            if on_tick is not None:
                try:
                    on_tick(record)
                except Exception:
                    log.exception("on_tick callback raised — continuing loop regardless")
        except Exception:
            log.exception("Unhandled error in run_tick — skipping this cycle, loop continues")
        cycles += 1
        elapsed = time.monotonic() - tick_start
        time.sleep(max(0.0, config.LIVE_TICK_INTERVAL_SECONDS - elapsed))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="run a single tick and exit")
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--buffer-bars", type=int, default=DEFAULT_BUFFER_BARS)
    parser.add_argument("--no-trading", action="store_true", help="run inference only, skip the paper-trading engine")
    args = parser.parse_args()

    if args.once:
        state = LiveLoopState(buffer_bars=args.buffer_bars, enable_trading=not args.no_trading)
        record = run_tick(state)
        log.info("Single tick complete: %s", record)
    else:
        run_forever(buffer_bars=args.buffer_bars, enable_trading=not args.no_trading, max_cycles=args.max_cycles)


if __name__ == "__main__":
    main()
