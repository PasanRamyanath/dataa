"""
Per-member checkpoint loaders for inference/live_loop.py (§12.1).

None of the 9 ensemble members are trained yet (see models/train_all.py —
training itself is out of scope here per explicit user instruction). Every
loader in this module is therefore written to the checkpoint format each
member's own train.py actually produces (confirmed by reading each train.py
directly), but returns None — logged once, not raised — when the checkpoint
file/directory isn't present on disk yet. inference/live_loop.py treats a
None loader result exactly like models/train_all.py treats a failed member:
skip it for this run, keep going with whichever members ARE available, and
report which ones are missing rather than crashing the whole loop. This is
what makes it safe to run live_loop.py today, before any Kaggle training run
has produced real checkpoints, and to keep running unattended as members
finish training one at a time.

Every loader wraps its member in a common `MemberPredictor` interface so
live_loop.py doesn't need to know each member's own input shape/library:

    predictor.predict(recent_df: pl.DataFrame) -> np.ndarray[3] | None

`recent_df` is the trailing window of feature-engineered rows (ascending by
ts) live_loop.py has assembled for the current tick — how many trailing rows
a given member actually needs (1 row for the two GBM members, LOOKBACK_PERIODS
for ssm-challenger, up to MAX_CONTEXT for kronos-finetune, input_size for the
neuralforecast members) is each predictor's own concern. Returns a (3,)
array [prob_down, prob_flat, prob_up] for the LAST row of recent_df, or None
if recent_df doesn't have enough history yet or the forward pass itself
raised (logged, not propagated — one member's bug must never take down the
whole live loop, same fault-isolation philosophy as models/train_all.py's
per-member subprocess isolation).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import numpy as np
import polars as pl

import config
from common.feature_contract import FEATURE_COLUMNS
from common.oof import DIRECTION_MEMBERS, regression_to_direction_proba

log = logging.getLogger("inference.checkpoint_loaders")

MODELS_ROOT = Path(__file__).resolve().parent.parent / "models"


class MemberPredictor(Protocol):
    name: str

    def predict(self, recent_df: pl.DataFrame) -> np.ndarray | None: ...


def _last_row_matrix(recent_df: pl.DataFrame, feature_cols: list[str]) -> np.ndarray:
    row = recent_df.tail(1)
    if any(c.startswith("news_") and c not in row.columns for c in feature_cols):
        # checkpoint was trained --with-news: join the same point-in-time
        # news block onto the current bar (scored store is cached per file
        # size, so this is cheap on every tick)
        from news_pipeline.features import attach_news_features

        row = attach_news_features(row)
    row = row.select(feature_cols).with_columns(
        [pl.col(c).cast(pl.Float32).fill_null(0.0) for c in feature_cols]
    )
    return row.to_numpy().astype(np.float32)


def _trained_feature_columns(checkpoint_dir: Path, default: list[str]) -> list[str]:
    """The exact column list a checkpoint was trained on (metrics.json's
    "feature_columns"), falling back to `default` for older checkpoints."""
    import json

    try:
        cols = json.loads((checkpoint_dir / "metrics.json").read_text(encoding="utf-8")).get("feature_columns")
    except (OSError, ValueError):
        cols = None
    return list(cols) if cols else list(default)


# ---------------------------------------------------------------------------
# gbm-family — LightGBM direction classifier, single-row tabular input.
# ---------------------------------------------------------------------------
class _GbmFamilyPredictor:
    name = "gbm-family"

    def __init__(self, booster, feature_cols: list[str] | None = None):
        self.booster = booster
        self.feature_cols = feature_cols or FEATURE_COLUMNS

    def predict(self, recent_df: pl.DataFrame) -> np.ndarray | None:
        if recent_df.height < 1:
            return None
        X = _last_row_matrix(recent_df, self.feature_cols)
        return self.booster.predict(X)[0]


def load_gbm_family() -> "_GbmFamilyPredictor | None":
    path = MODELS_ROOT / "gbm_family" / "checkpoints" / "direction_classifier.txt"
    if not path.exists():
        log.info("gbm-family: no checkpoint at %s — member unavailable this run", path)
        return None
    try:
        import lightgbm as lgb

        return _GbmFamilyPredictor(lgb.Booster(model_file=str(path)),
                                   _trained_feature_columns(path.parent, FEATURE_COLUMNS))
    except Exception:
        log.exception("gbm-family: failed to load checkpoint at %s", path)
        return None


# ---------------------------------------------------------------------------
# sentiment-branch — LightGBM over SENTIMENT_PROXY_FEATURES (crowd-positioning
# proxies) and, when trained --with-news, news_pipeline's text-sentiment
# block too (column list read from the checkpoint's metrics.json).
# ---------------------------------------------------------------------------
SENTIMENT_PROXY_FEATURES = [
    "funding_rate", "funding_rate_ma_8h", "funding_rate_ma_24h", "funding_rate_change",
    "toptrader_count_ls_ratio", "toptrader_sum_ls_ratio", "all_account_ls_ratio",
    "taker_ls_vol_ratio", "funding_oi_crowding_180", "funding_oi_crowding_720",
    "implied_vol", "implied_vol_change_1", "iv_realized_vol_spread",
]


class _SentimentBranchPredictor:
    name = "sentiment-branch"

    def __init__(self, booster, feature_cols: list[str] | None = None):
        self.booster = booster
        self.feature_cols = feature_cols or SENTIMENT_PROXY_FEATURES

    def predict(self, recent_df: pl.DataFrame) -> np.ndarray | None:
        if recent_df.height < 1:
            return None
        X = _last_row_matrix(recent_df, self.feature_cols)
        return self.booster.predict(X)[0]


def load_sentiment_branch() -> "_SentimentBranchPredictor | None":
    path = MODELS_ROOT / "sentiment_branch" / "checkpoints" / "sentiment_proxy_classifier.txt"
    if not path.exists():
        log.info("sentiment-branch: no checkpoint at %s — member unavailable this run", path)
        return None
    try:
        import lightgbm as lgb

        return _SentimentBranchPredictor(lgb.Booster(model_file=str(path)),
                                         _trained_feature_columns(path.parent, SENTIMENT_PROXY_FEATURES))
    except Exception:
        log.exception("sentiment-branch: failed to load checkpoint at %s", path)
        return None


# ---------------------------------------------------------------------------
# ssm-challenger — torch Mamba/GRU sequence model, needs a full LOOKBACK_PERIODS
# window (common/windowing.py's convention: window i covers rows [i-lookback+1, i]).
# ---------------------------------------------------------------------------
class _SsmChallengerPredictor:
    name = "ssm-challenger"

    def __init__(self, model, torch_module, device: str, feature_cols: list[str] | None = None,
                 scaler: dict[str, np.ndarray] | None = None, lookback: int | None = None):
        self.model = model
        self.lookback = lookback or config.LOOKBACK_PERIODS
        self.torch = torch_module
        self.device = device
        self.feature_cols = feature_cols or FEATURE_COLUMNS
        self.scaler = scaler

    def window(self, recent_df: pl.DataFrame) -> np.ndarray | None:
        """(1, lookback, n_features) input exactly as training built it:
        the checkpoint's own column list and train-window standardization."""
        from common.windowing import apply_feature_scaler, feature_matrix

        lookback = self.lookback
        if recent_df.height < lookback:
            return None
        return apply_feature_scaler(feature_matrix(recent_df.tail(lookback), self.feature_cols), self.scaler)[None, :, :]

    def predict(self, recent_df: pl.DataFrame) -> np.ndarray | None:
        X = self.window(recent_df)
        if X is None:
            return None
        with self.torch.no_grad():
            x_t = self.torch.from_numpy(X).to(self.device)
            direction_logits, _return_pred = self.model(x_t)
            proba = self.torch.softmax(direction_logits, dim=-1).cpu().numpy()[0]
        return proba


def load_ssm_challenger() -> "_SsmChallengerPredictor | None":
    path = MODELS_ROOT / "ssm_challenger" / "checkpoints" / "ssm_challenger.pt"
    if not path.exists():
        log.info("ssm-challenger: no checkpoint at %s — member unavailable this run", path)
        return None
    try:
        import torch

        from models.ssm_challenger.model import SSMChallenger

        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(path, map_location=device)
        model = SSMChallenger(n_features=ckpt["n_features"])
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()
        scaler = None
        if "feature_mean" in ckpt:
            scaler = {"mean": ckpt["feature_mean"].cpu().numpy(), "scale": ckpt["feature_scale"].cpu().numpy()}
        else:
            log.warning("ssm-challenger: checkpoint predates input scaling — retrain it; feeding raw features")
        return _SsmChallengerPredictor(model, torch, device, ckpt.get("feature_columns"), scaler, ckpt.get("lookback"))
    except Exception:
        log.exception("ssm-challenger: failed to load checkpoint at %s", path)
        return None


# ---------------------------------------------------------------------------
# NeuralForecast members (tft, patchtst, itransformer, nhits) — share
# common/neuralforecast_harness.py's to_long_format() convention and produce
# a raw price-path forecast, converted to direction proba via
# common.oof.regression_to_direction_proba (same convention
# models/{tft,patchtst,itransformer,nhits}/train.py's own OOF export uses).
# ---------------------------------------------------------------------------
class _NeuralForecastPredictor:
    def __init__(self, name: str, nf):
        self.name = name
        self.nf = nf

    @staticmethod
    def future_frame(recent_df: pl.DataFrame, h: int, futr_cols: list[str]):
        """The h bars after the latest one with their calendar/funding-clock
        columns — a model trained with futr_exog_list (TFT, N-HiTS) refuses
        to predict without them. feature_engine.temporal computes them from
        the timestamp alone, the same code build_dataset used in training."""
        from datetime import timedelta

        from common.neuralforecast_harness import UNIQUE_ID
        from feature_engine.temporal import add_temporal_features

        last_ts = recent_df["ts"][-1]
        step = timedelta(seconds=config.PRIMARY_INTERVAL_SECONDS)
        fut = pl.DataFrame({"ts": pl.datetime_range(last_ts + step, last_ts + step * h, step, eager=True,
                                                      time_unit=recent_df.schema["ts"].time_unit,
                                                      time_zone=recent_df.schema["ts"].time_zone)})
        fut = add_temporal_features(fut)
        return fut.select(
            pl.col("ts").alias("ds"),
            *[pl.col(c).cast(pl.Float32) for c in futr_cols],
        ).with_columns(pl.lit(UNIQUE_ID).alias("unique_id")).to_pandas()

    def predict(self, recent_df: pl.DataFrame) -> np.ndarray | None:
        from common.neuralforecast_harness import to_long_format

        if recent_df.height < config.LOOKBACK_PERIODS:
            return None
        model = self.nf.models[0]
        hist = list(getattr(model, "hist_exog_list", None) or [])
        futr = list(getattr(model, "futr_exog_list", None) or [])
        long_df = to_long_format(recent_df, hist + futr)
        futr_df = self.future_frame(recent_df, int(model.h), futr) if futr else None
        forecast = self.nf.predict(df=long_df, futr_df=futr_df)
        if "unique_id" not in forecast.columns:
            forecast = forecast.reset_index()
        pred_cols = [c for c in forecast.columns if c not in ("unique_id", "ds", "index")]
        if not pred_cols:
            return None
        # End-of-horizon forecast vs the latest close — the same signal
        # common/neuralforecast_harness.py exported as this member's OOF.
        last_known_close = float(recent_df.tail(1)["close"][0])
        y_pred_horizon_end = float(forecast[pred_cols[0]].iloc[-1])
        pred_return = np.array([(y_pred_horizon_end - last_known_close) / last_known_close])
        return regression_to_direction_proba(pred_return)[0]


def _load_neuralforecast_member(member_dir: str, member_name: str) -> "_NeuralForecastPredictor | None":
    path = MODELS_ROOT / member_dir / "checkpoints"
    if not path.exists() or not any(path.iterdir()):
        log.info("%s: no checkpoint directory at %s — member unavailable this run", member_name, path)
        return None
    try:
        from neuralforecast import NeuralForecast

        nf = NeuralForecast.load(path=str(path))
        return _NeuralForecastPredictor(member_name, nf)
    except Exception:
        log.exception("%s: failed to load checkpoint at %s", member_name, path)
        return None


def load_tft() -> "_NeuralForecastPredictor | None":
    return _load_neuralforecast_member("tft", "tft")


def load_patchtst() -> "_NeuralForecastPredictor | None":
    return _load_neuralforecast_member("patchtst", "patchtst")


def load_itransformer() -> "_NeuralForecastPredictor | None":
    return _load_neuralforecast_member("itransformer", "itransformer")


def load_nhits() -> "_NeuralForecastPredictor | None":
    return _load_neuralforecast_member("nhits", "nhits")


# ---------------------------------------------------------------------------
# kronos-finetune — vendored Kronos repo, OHLCV-only context (not the full
# 184-column FEATURE_COLUMNS), capped at MAX_CONTEXT bars.
# ---------------------------------------------------------------------------
MAX_CONTEXT = 512
KRONOS_PREDICT_LEN = 48


class _KronosPredictor:
    name = "kronos-finetune"

    def __init__(self, predictor):
        self.predictor = predictor

    def predict(self, recent_df: pl.DataFrame) -> np.ndarray | None:
        # kronos_finetune/train.py's OOF windows always carry a full
        # MAX_CONTEXT of history; a shorter warm-up context is a different
        # input distribution than the stacking meta-learner was fit on.
        if recent_df.height < MAX_CONTEXT:
            return None
        ctx = recent_df.tail(MAX_CONTEXT)
        import pandas as pd

        x_df = ctx.select(["open", "high", "low", "close", "volume"]).to_pandas()
        x_ts = ctx["ts"].to_pandas()
        step = config.PRIMARY_INTERVAL_SECONDS
        last_ts = ctx.tail(1)["ts"].to_pandas().iloc[0]
        y_ts = pd.Series(pd.date_range(start=last_ts, periods=KRONOS_PREDICT_LEN + 1, freq=f"{step}s")[1:])

        forecast = self.predictor.predict(
            df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
            pred_len=KRONOS_PREDICT_LEN, T=1.0, top_p=0.9, sample_count=1,
        )
        if forecast is None or len(forecast) == 0 or "close" not in forecast.columns:
            return None
        # End-of-horizon vs latest close, as kronos_finetune/train.py's OOF.
        last_known_close = float(ctx.tail(1)["close"][0])
        y_pred_horizon_end = float(forecast["close"].iloc[-1])
        pred_return = np.array([(y_pred_horizon_end - last_known_close) / last_known_close])
        return regression_to_direction_proba(pred_return)[0]


def load_kronos_finetune() -> "_KronosPredictor | None":
    base = MODELS_ROOT / "kronos_finetune"
    exp_name = f"{config.SYMBOL_UM}_{config.PRIMARY_INTERVAL}"
    finetuned_tok = base / "finetuned" / exp_name / "tokenizer" / "best_model"
    finetuned_model = base / "finetuned" / exp_name / "basemodel" / "best_model"
    pretrained_tok = base / "pretrained" / "Kronos-Tokenizer-base"
    pretrained_model = base / "pretrained" / "Kronos-base"

    if finetuned_tok.exists() and finetuned_model.exists():
        tok_path, model_path = finetuned_tok, finetuned_model
    elif pretrained_tok.exists() and pretrained_model.exists():
        tok_path, model_path = pretrained_tok, pretrained_model
        log.info("kronos-finetune: no finetuned checkpoint — falling back to zero-shot pretrained weights")
    else:
        log.info("kronos-finetune: no tokenizer/model checkpoint under %s — member unavailable this run", base)
        return None

    vendor_dir = base / "vendor" / "Kronos"
    if not vendor_dir.exists():
        log.info("kronos-finetune: vendored repo not present at %s — member unavailable this run", vendor_dir)
        return None

    try:
        import sys

        if str(vendor_dir) not in sys.path:
            sys.path.insert(0, str(vendor_dir))
        from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: F401 (Kronos imported for parity with train.py)

        tokenizer = KronosTokenizer.from_pretrained(str(tok_path))
        model = Kronos.from_pretrained(str(model_path))
        predictor = KronosPredictor(model, tokenizer, max_context=MAX_CONTEXT)
        return _KronosPredictor(predictor)
    except Exception:
        log.exception("kronos-finetune: failed to load checkpoint (tokenizer=%s, model=%s)", tok_path, model_path)
        return None


# ---------------------------------------------------------------------------
# hmm-regime — gating input, not a directional vote. Returns a plain int
# state (not a (3,) proba array) via predict_regime(), kept separate from
# MemberPredictor/load_all_members() below since fusion.build_meta_features
# treats it as a one-hot gating column, never as a member OOF vote (see
# common/oof.py's load_all_oof() stem filter excluding "regime_states").
# ---------------------------------------------------------------------------
class HmmRegimePredictor:
    name = "hmm-regime"

    def __init__(self, model, mean: np.ndarray, std: np.ndarray, features: list[str], n_states: int):
        self.model = model
        self.mean = mean
        self.std = std
        self.features = features
        self.n_states = n_states

    def predict_regime(self, recent_df: pl.DataFrame, context_bars: int = 360) -> int | None:
        if recent_df.height < 1:
            return None
        window = recent_df.tail(min(context_bars, recent_df.height)).select(self.features).with_columns(
            [pl.col(c).cast(pl.Float64).fill_null(0.0) for c in self.features]
        )
        X = window.to_numpy()
        X_norm = (X - self.mean) / self.std
        from models.hmm_regime.decode import filtered_states

        # Forward filter, identical to how training exported the OOF regime
        # states the meta-learner was fit on.
        return int(filtered_states(self.model, X_norm)[-1])

    def regime_volatility_rank(self, state: int) -> float:
        """Returns state's mean realized_vol_90 (raw HMM-fitted mean, not
        normalized) — used by paper_trading_engine.regime_multiplier() to
        rank fitted states by volatility without assuming a fixed state
        index meaning (see config.py's DEFAULT/LOW/HIGH_VOL_REGIME_SIZE_MULTIPLIER
        docstring for why: hmmlearn doesn't guarantee state 0 == low-vol)."""
        if "realized_vol_90" not in self.features:
            return 0.0
        idx = self.features.index("realized_vol_90")
        return float(self.model.means_[state, idx])


def load_hmm_regime() -> "HmmRegimePredictor | None":
    path = MODELS_ROOT / "hmm_regime" / "checkpoints" / "hmm_regime_model.pkl"
    if not path.exists():
        log.info("hmm-regime: no checkpoint at %s — regime gating unavailable this run", path)
        return None
    try:
        import pickle

        with open(path, "rb") as f:
            state = pickle.load(f)
        return HmmRegimePredictor(
            model=state["model"], mean=state["mean"], std=state["std"],
            features=state["features"], n_states=state["n_states"],
        )
    except Exception:
        log.exception("hmm-regime: failed to load checkpoint at %s", path)
        return None


DIRECTION_MEMBER_LOADERS = {
    "gbm-family": load_gbm_family,
    "sentiment-branch": load_sentiment_branch,
    "ssm-challenger": load_ssm_challenger,
    "tft": load_tft,
    "patchtst": load_patchtst,
    "itransformer": load_itransformer,
    "nhits": load_nhits,
    "kronos-finetune": load_kronos_finetune,
}
assert set(DIRECTION_MEMBER_LOADERS) == set(DIRECTION_MEMBERS),     "common.oof.DIRECTION_MEMBERS (what fusion trains on) must list exactly the live members"


# ---------------------------------------------------------------------------
# Fusion stage (fusion/stacking_meta_learner.py, fusion/calibration.py,
# fusion/meta_labeling.py) — each wraps a small pickled sklearn object; see
# those modules' own load() classmethods for the exact payload shape.
# ---------------------------------------------------------------------------
def load_stacking_meta_learner():
    from fusion.stacking_meta_learner import CHECKPOINT_DIR, StackingMetaLearner

    path = CHECKPOINT_DIR / "stacking_meta_learner.pkl"
    if not path.exists():
        log.info("fusion stacking meta-learner: no checkpoint at %s — fusion stage unavailable this run", path)
        return None
    try:
        return StackingMetaLearner.load(path)
    except Exception:
        log.exception("fusion stacking meta-learner: failed to load checkpoint at %s", path)
        return None


def load_calibrator():
    from fusion.calibration import CHECKPOINT_DIR, PerClassCalibrator

    path = CHECKPOINT_DIR / "per_class_calibrator.pkl"
    if not path.exists():
        log.info("fusion calibrator: no checkpoint at %s — predictions will be uncalibrated", path)
        return None
    try:
        return PerClassCalibrator.load(path)
    except Exception:
        log.exception("fusion calibrator: failed to load checkpoint at %s", path)
        return None


def load_meta_labeling_filter():
    from fusion.meta_labeling import CHECKPOINT_DIR, MetaLabelFilter

    path = CHECKPOINT_DIR / "meta_labeling_filter.pkl"
    if not path.exists():
        log.info("fusion meta-labeling filter: no checkpoint at %s — no act/bet_size gating this run", path)
        return None
    try:
        return MetaLabelFilter.load(path)
    except Exception:
        log.exception("fusion meta-labeling filter: failed to load checkpoint at %s", path)
        return None


def load_all_members() -> dict[str, "MemberPredictor"]:
    """Loads every direction-voting member whose checkpoint currently
    exists on disk. Missing/broken members are logged and simply absent
    from the returned dict — live_loop.py must tolerate any subset,
    including the empty dict (no members trained yet)."""
    members = {}
    for name, loader in DIRECTION_MEMBER_LOADERS.items():
        predictor = loader()
        if predictor is not None:
            members[name] = predictor
    log.info("Loaded %d/%d direction-voting members: %s", len(members), len(DIRECTION_MEMBER_LOADERS), sorted(members))
    return members
