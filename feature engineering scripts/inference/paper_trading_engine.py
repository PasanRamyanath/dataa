"""
Paper-trading engine per plan §12.2 — confidence-scaled position sizing with
a regime multiplier, daily/weekly circuit breakers, and full trade logging.
No live-execution path exists anywhere in this module: it only ever marks a
simulated position to the close price inference/live_loop.py hands it and
writes JSONL records to config.PAPER_TRADING_DATA_ROOT. Nothing here ever
calls an exchange API, places a real order, or touches real funds.

Sizing model (documented interpretation — §12.2 specifies "confidence-scaled
... times a regime multiplier" without a formula): target position fraction
of equity =

    sign(primary_direction) * confidence**2 * bet_size * regime_multiplier

clipped to +/- config.PAPER_MAX_POSITION_FRACTION. `confidence` and
`bet_size` come from fusion.meta_labeling.MetaLabelFilter's act_probability/
bet_size (via inference/live_loop.py's fusion stage) — confidence is
squared per §12.2's literal wording ("confidence-scaled") interpreted as a
convex (not linear) scaling so only genuinely confident calls get
meaningfully sized, matching the plan's general caution around over-trading
low-conviction signals. `regime_multiplier` ranks the HMM's *fitted* states
by their own realized_vol_90 mean (see inference/checkpoint_loaders.py's
HmmRegimePredictor.regime_volatility_rank) into low/mid/high vol terciles —
never by raw state index, since hmmlearn doesn't guarantee state 0 means
anything in particular.

Position tracking uses a continuous-rebalancing model (not discrete
entry/exit order tracking): the engine holds one signed `position_fraction`
of equity at a time, marks it to market every tick, and pays simulated
fee+slippage cost on any change in that fraction. This is a deliberate
simplification — the plan doesn't specify an order-book-level fill model —
appropriate for judging whether the ensemble's directional calls are
profitable net of realistic trading costs, not for simulating microstructure.

Circuit breakers (§12.2) halt NEW entries once tripped — they never force-
close an existing position — and reset automatically at the next UTC day/
ISO-week boundary. "New entry" here means growing |position_fraction| or
opening a position from flat/flipping sign; a tripped breaker still allows
the position to shrink toward flat.

Usage:
    from inference.paper_trading_engine import PaperTradingEngine
    engine = PaperTradingEngine()
    engine.on_tick(ts=..., close_price=..., fused=fused_dict, regime_state=..., hmm=hmm_predictor)

    python -m inference.paper_trading_engine --demo   # synthetic self-check, no real checkpoints needed
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("inference.paper_trading_engine")

_DIRECTION_SIGN = {0: -1.0, 1: 0.0, 2: 1.0}  # {down, flat, up} -> {-1, 0, +1}


def regime_multiplier(regime_state: int | None, hmm) -> float:
    """Ranks the HMM's fitted states by realized_vol_90 mean into terciles
    (low/mid/high vol) and returns the corresponding config multiplier.
    Falls back to the neutral default when regime gating isn't available
    this tick (hmm-regime checkpoint missing, or fewer than 2 states)."""
    if regime_state is None or hmm is None or getattr(hmm, "n_states", 0) < 2:
        return config.DEFAULT_REGIME_SIZE_MULTIPLIER
    vols = [hmm.regime_volatility_rank(s) for s in range(hmm.n_states)]
    order = list(np.argsort(vols))  # ascending volatility
    rank = order.index(regime_state)
    frac = rank / max(hmm.n_states - 1, 1)
    if frac <= 1 / 3:
        return config.LOW_VOL_REGIME_SIZE_MULTIPLIER
    if frac >= 2 / 3:
        return config.HIGH_VOL_REGIME_SIZE_MULTIPLIER
    return config.DEFAULT_REGIME_SIZE_MULTIPLIER


def _utc_day_key(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%d")


def _utc_week_key(ts: datetime) -> str:
    iso = ts.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


class PaperTradingEngine:
    def __init__(self, data_root: Path | None = None, starting_equity: float | None = None):
        self.data_root = data_root or config.PAPER_TRADING_DATA_ROOT
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.data_root / "engine_state.json"
        self.trade_log_path = self.data_root / "trades.jsonl"
        self.equity_curve_path = self.data_root / "equity_curve.jsonl"

        self.equity = starting_equity if starting_equity is not None else config.PAPER_STARTING_EQUITY_USD
        self.position_fraction = 0.0
        self.last_price: float | None = None
        self.day_key: str | None = None
        self.day_start_equity = self.equity
        self.daily_breaker_tripped = False
        self.week_key: str | None = None
        self.week_start_equity = self.equity
        self.weekly_breaker_tripped = False
        self.n_ticks = 0
        self.n_trades = 0

        self._load_state()

    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            log.warning("Corrupt engine state at %s — starting fresh with default equity", self.state_path)
            return
        self.equity = state.get("equity", self.equity)
        self.position_fraction = state.get("position_fraction", 0.0)
        self.last_price = state.get("last_price")
        self.day_key = state.get("day_key")
        self.day_start_equity = state.get("day_start_equity", self.equity)
        self.daily_breaker_tripped = state.get("daily_breaker_tripped", False)
        self.week_key = state.get("week_key")
        self.week_start_equity = state.get("week_start_equity", self.equity)
        self.weekly_breaker_tripped = state.get("weekly_breaker_tripped", False)
        self.n_ticks = state.get("n_ticks", 0)
        self.n_trades = state.get("n_trades", 0)
        log.info("Restored paper-trading state from %s: equity=%.2f position_fraction=%.4f", self.state_path, self.equity, self.position_fraction)

    def _save_state(self) -> None:
        state = {
            "equity": self.equity, "position_fraction": self.position_fraction, "last_price": self.last_price,
            "day_key": self.day_key, "day_start_equity": self.day_start_equity, "daily_breaker_tripped": self.daily_breaker_tripped,
            "week_key": self.week_key, "week_start_equity": self.week_start_equity, "weekly_breaker_tripped": self.weekly_breaker_tripped,
            "n_ticks": self.n_ticks, "n_trades": self.n_trades,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def _roll_day_week(self, ts: datetime) -> None:
        day_key = _utc_day_key(ts)
        if self.day_key != day_key:
            self.day_key = day_key
            self.day_start_equity = self.equity
            self.daily_breaker_tripped = False
        week_key = _utc_week_key(ts)
        if self.week_key != week_key:
            self.week_key = week_key
            self.week_start_equity = self.equity
            self.weekly_breaker_tripped = False

    def _check_circuit_breakers(self) -> None:
        if self.day_start_equity > 0:
            daily_dd = (self.day_start_equity - self.equity) / self.day_start_equity
            if daily_dd >= config.PAPER_DAILY_MAX_DRAWDOWN_PCT and not self.daily_breaker_tripped:
                self.daily_breaker_tripped = True
                log.warning("Daily circuit breaker TRIPPED: drawdown %.2f%% >= %.2f%%", daily_dd * 100, config.PAPER_DAILY_MAX_DRAWDOWN_PCT * 100)
        if self.week_start_equity > 0:
            weekly_dd = (self.week_start_equity - self.equity) / self.week_start_equity
            if weekly_dd >= config.PAPER_WEEKLY_MAX_DRAWDOWN_PCT and not self.weekly_breaker_tripped:
                self.weekly_breaker_tripped = True
                log.warning("Weekly circuit breaker TRIPPED: drawdown %.2f%% >= %.2f%%", weekly_dd * 100, config.PAPER_WEEKLY_MAX_DRAWDOWN_PCT * 100)

    def _clamp_for_breakers(self, target_fraction: float) -> float:
        if not (self.daily_breaker_tripped or self.weekly_breaker_tripped):
            return target_fraction
        current = self.position_fraction
        same_or_flat = (np.sign(target_fraction) == np.sign(current)) or target_fraction == 0.0
        is_reduction = abs(target_fraction) <= abs(current)
        if same_or_flat and is_reduction:
            return target_fraction
        log.info("Circuit breaker active — blocking new entry (target=%.4f, holding at %.4f)", target_fraction, current)
        return current

    def compute_target_fraction(self, fused: dict, regime_state: int | None, hmm) -> float | None:
        primary_direction = fused.get("primary_direction")
        confidence = fused.get("confidence")
        bet_size = fused.get("bet_size")
        act_probability = fused.get("act_probability")
        if primary_direction is None or confidence is None or bet_size is None:
            return None  # meta-labeling gate unavailable this tick — no sizing decision possible

        # Hard-zero conviction veto per Lopez de Prado meta-labeling architecture:
        # If the meta-label filter predicts conviction below the action threshold
        # (or bet_size is zero), veto entering or holding directional exposure.
        action_thresh = getattr(config, "META_LABEL_ACTION_THRESHOLD", 0.5)
        if (act_probability is not None and act_probability < action_thresh) or float(bet_size) <= 0.0:
            return 0.0

        sign = _DIRECTION_SIGN.get(int(primary_direction), 0.0)
        raw = sign * (float(confidence) ** 2) * float(bet_size) * regime_multiplier(regime_state, hmm)
        return float(np.clip(raw, -config.PAPER_MAX_POSITION_FRACTION, config.PAPER_MAX_POSITION_FRACTION))

    def on_tick(self, ts: datetime, close_price: float, fused: dict, regime_state: int | None = None, hmm=None) -> dict:
        self.n_ticks += 1
        self._roll_day_week(ts)

        if self.last_price is not None and self.position_fraction != 0.0:
            price_return = (close_price - self.last_price) / self.last_price
            pnl = self.equity * self.position_fraction * price_return
            self.equity += pnl

        target_fraction = self.compute_target_fraction(fused, regime_state, hmm)
        trade_record = None
        if target_fraction is not None:
            target_fraction = self._clamp_for_breakers(target_fraction)
            min_rebal = getattr(config, "PAPER_MIN_REBALANCE_FRACTION", 0.01)
            diff = abs(target_fraction - self.position_fraction)
            # Rebalance only if position delta exceeds deadband threshold (1% of equity)
            # or if explicitly flattening the position (target_fraction == 0.0)
            if diff >= min_rebal or (target_fraction == 0.0 and self.position_fraction != 0.0):
                turnover = diff
                cost = self.equity * turnover * (config.PAPER_TAKER_FEE_BPS + config.PAPER_SLIPPAGE_BPS) / 10_000.0
                self.equity -= cost
                self.n_trades += 1
                trade_record = {
                    "ts": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                    "price": close_price, "old_fraction": self.position_fraction,
                    "new_fraction": target_fraction, "turnover": turnover, "cost_usd": cost,
                    "equity_after": self.equity, "regime_state": regime_state,
                }
                with open(self.trade_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(trade_record, default=str) + "\n")
                self.position_fraction = target_fraction

        self.last_price = close_price
        self._check_circuit_breakers()
        self._save_state()

        equity_record = {
            "ts": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
            "equity": self.equity, "position_fraction": self.position_fraction,
            "daily_breaker_tripped": self.daily_breaker_tripped, "weekly_breaker_tripped": self.weekly_breaker_tripped,
        }
        with open(self.equity_curve_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(equity_record, default=str) + "\n")

        return {"trade": trade_record, **equity_record}


def load_latest_engine_state(data_root: Path | None = None) -> dict | None:
    """Read-only accessor for a streaming/dashboard layer (Gap 6) to report
    current equity/position without instantiating a full engine."""
    path = (data_root or config.PAPER_TRADING_DATA_ROOT) / "engine_state.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _demo(n_ticks: int) -> None:
    """Synthetic self-check: a random-walk price series with a mildly
    informative random signal, run through the full sizing/circuit-breaker/
    logging path — no real checkpoints or network access required."""
    import shutil
    import tempfile

    rng = np.random.default_rng(7)
    tmp_dir = Path(tempfile.mkdtemp(prefix="paper_trading_demo_"))
    engine = PaperTradingEngine(data_root=tmp_dir, starting_equity=100_000.0)
    price = 60_000.0
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    from datetime import timedelta

    for i in range(n_ticks):
        price *= 1.0 + rng.normal(0, 0.0006)
        ts = ts + timedelta(seconds=10)
        direction = rng.choice([0, 1, 2], p=[0.3, 0.4, 0.3])
        confidence = float(rng.uniform(0.05, 0.9))
        bet_size = float(rng.uniform(0.1, 1.0))
        fused = {"primary_direction": direction, "confidence": confidence, "bet_size": bet_size}
        regime_state = int(rng.integers(0, 3))

        class _FakeHmm:
            n_states = 3

            def regime_volatility_rank(self, s):
                return [0.001, 0.005, 0.02][s]

        result = engine.on_tick(ts=ts, close_price=price, fused=fused, regime_state=regime_state, hmm=_FakeHmm())
        if i % 200 == 0:
            log.info("tick %d: price=%.2f equity=%.2f position_fraction=%.4f", i, price, result["equity"], result["position_fraction"])

    log.info("Demo complete: %d ticks, %d trades, final equity=%.2f (started at 100000.00)", engine.n_ticks, engine.n_trades, engine.equity)
    shutil.rmtree(tmp_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="run a synthetic self-check simulation and exit")
    parser.add_argument("--demo-ticks", type=int, default=2000)
    args = parser.parse_args()
    if args.demo:
        _demo(args.demo_ticks)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
