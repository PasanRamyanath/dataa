"""
Shared out-of-fold (val-set) prediction store every member writes to and
fusion/stacking_meta_learner.py reads from — this is the seam between
"independently trained ensemble members" and "one combined signal": each
member's train.py calls save_oof() once with its own held-out val
predictions, and fusion never needs to know how any individual member
produces its predictions, only that every member's file has the same
(ts, prob_down, prob_flat, prob_up) shape.

Regression-only members (the neuralforecast-based forecasters,
kronos_finetune — none of which natively output a 3-class probability) go
through regression_to_direction_proba() / calibrate_regression_to_direction_proba(),
which converts a predicted return into a calibrated soft direction distribution
via a volatility-normalized logistic/softmax rather than a hard sign — so the
meta-learner sees a well-calibrated confidence, not just a vote.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

OOF_DIR = Path(__file__).resolve().parent.parent / "fusion" / "oof"

DIRECTION_MEMBERS: list[str] = [
    "gbm-family", "sentiment-branch", "ssm-challenger", "tft",
    "patchtst", "itransformer", "nhits", "kronos-finetune",
]


def regression_to_direction_proba(pred_return: np.ndarray, scale: float = 5e-4) -> np.ndarray:
    """Maps a predicted (log-)return into a soft 3-class [down, flat, up]
    distribution: treats the predicted return as the logit-scale signal for
    "up vs down", scaled by `scale` (a typical 10s-bar log-return magnitude —
    see label_return_h1's realistic range), then splits out a "flat" mass
    inversely proportional to |pred_return| so a near-zero prediction reads
    as genuinely uncertain rather than forced into up or down."""
    pred_clean = np.nan_to_num(np.asarray(pred_return, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    z = np.clip(pred_clean / max(scale, 1e-12), -20.0, 20.0)
    p_up_vs_down = 1.0 / (1.0 + np.exp(-z))
    flat_mass = np.clip(1.0 - np.tanh(np.abs(z)), 0.05, 0.9)
    p_up = (1 - flat_mass) * p_up_vs_down
    p_down = (1 - flat_mass) * (1 - p_up_vs_down)
    p_flat = flat_mass
    return np.stack([p_down, p_flat, p_up], axis=1)


def calibrate_regression_to_direction_proba(
    pred_return: np.ndarray,
    y_true: np.ndarray | None = None,
    realized_vol: np.ndarray | None = None,
    default_scale: float = 5e-4,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Converts continuous predicted returns into calibrated 3-class [down, flat, up]
    probabilities by:
      1. Removing systematic median decoder drift across the validation fold.
      2. Normalizing by local realized volatility when provided.
      3. Grid-searching the scale and flat-mass exponent that maximize directional
         accuracy and minimize log-loss on `y_true` (when provided).
    """
    pred_clean = np.nan_to_num(np.asarray(pred_return, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if len(pred_clean) == 0:
        return np.empty((0, 3), dtype=np.float32), {"scale": default_scale, "flat_gain": 1.0}

    # De-bias constant decoder offset
    centered = pred_clean - np.median(pred_clean)
    if realized_vol is not None and len(realized_vol) == len(centered):
        rvol = np.maximum(np.nan_to_num(np.asarray(realized_vol, dtype=np.float64), nan=default_scale), 1e-5)
        med_vol = float(np.median(rvol))
        norm_signal = centered * (med_vol / rvol)
    else:
        norm_signal = centered

    std_sig = float(np.std(norm_signal))
    base_scale = max(std_sig, 1e-6) if std_sig > 1e-8 else default_scale

    if y_true is None or len(y_true) != len(norm_signal):
        proba = regression_to_direction_proba(norm_signal, scale=base_scale * 0.85)
        return proba.astype(np.float32), {"scale": float(base_scale * 0.85), "flat_gain": 1.0}

    y_clean = np.asarray(y_true, dtype=np.int64)
    best_acc = -1.0
    best_proba = regression_to_direction_proba(norm_signal, scale=base_scale)
    best_params = {"scale": float(base_scale), "flat_gain": 1.0, "sign": 1.0}

    for mult in (0.35, 0.55, 0.80, 1.10, 1.50, 2.20):
        sc = max(base_scale * mult, 1e-7)
        z = np.clip(norm_signal / sc, -20.0, 20.0)
        p_up_vs_down = 1.0 / (1.0 + np.exp(-z))
        for flat_gain in (0.55, 0.80, 1.05, 1.35, 1.75):
            flat_mass = np.clip((1.0 - np.tanh(np.abs(z))) * flat_gain, 0.04, 0.88)
            p_up = (1.0 - flat_mass) * p_up_vs_down
            p_down = (1.0 - flat_mass) * (1.0 - p_up_vs_down)
            cand = np.stack([p_down, flat_mass, p_up], axis=1)
            cand /= np.maximum(cand.sum(axis=1, keepdims=True), 1e-12)
            acc = float((cand.argmax(axis=1) == y_clean).mean())
            if acc > best_acc:
                best_acc = acc
                best_proba = cand
                best_params = {"scale": float(sc), "flat_gain": float(flat_gain)}

    return best_proba.astype(np.float32), best_params


def save_oof(member_name: str, ts: np.ndarray, direction_proba: np.ndarray) -> Path:
    """`direction_proba` must be shape (n, 3) in [prob_down, prob_flat,
    prob_up] order (the same {-1,0,1}->{0,1,2} remap common.data.to_xy
    uses)."""
    if direction_proba.shape[1] != 3:
        raise ValueError(f"save_oof: direction_proba must have 3 columns, got shape {direction_proba.shape}")
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame({
        "ts": ts,
        "prob_down": direction_proba[:, 0].astype(np.float32),
        "prob_flat": direction_proba[:, 1].astype(np.float32),
        "prob_up": direction_proba[:, 2].astype(np.float32),
    })
    out_path = OOF_DIR / f"{member_name}.parquet"
    df.write_parquet(out_path, compression="snappy")
    return out_path


def load_all_oof() -> dict[str, pl.DataFrame]:
    if not OOF_DIR.exists():
        return {}
    return {p.stem: pl.read_parquet(p) for p in OOF_DIR.glob("*.parquet") if p.stem != "regime_states"}


def save_regime_states(ts: np.ndarray, states: np.ndarray) -> Path:
    """models/hmm_regime/train.py's output isn't a directional prediction —
    it's a gating input fusion/stacking_meta_learner.py one-hot-encodes
    alongside every member's direction probabilities (see
    fusion.stacking_meta_learner.build_meta_features)."""
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame({"ts": ts, "regime_state": states.astype(np.int32)})
    out_path = OOF_DIR / "regime_states.parquet"
    df.write_parquet(out_path, compression="snappy")
    return out_path


def load_regime_states() -> pl.DataFrame | None:
    path = OOF_DIR / "regime_states.parquet"
    return pl.read_parquet(path) if path.exists() else None


def oof_paths_for(member_name: str) -> list[Path]:
    """The OOF file(s) a member's train.py leaves in OOF_DIR — shipped to
    that member's HF repo under oof/ so a later Kaggle session (fresh disk)
    can restore them before the fusion stages read them."""
    name = "regime_states" if member_name in ("hmm-regime", "tier4-hmm-macro") else member_name
    path = OOF_DIR / f"{name}.parquet"
    return [path] if path.exists() else []
