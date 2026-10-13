"""
Shared out-of-fold (val-set) prediction store every member writes to and
fusion/stacking_meta_learner.py reads from — this is the seam between
"9 independently trained ensemble members" and "one combined signal": each
member's train.py calls save_oof() once with its own held-out val
predictions, and fusion never needs to know how any individual member
produces its predictions, only that every member's file has the same
(ts, prob_down, prob_flat, prob_up) shape.

Regression-only members (the neuralforecast-based forecasters, ssm_challenger,
kronos_finetune — none of which natively output a 3-class probability) go
through regression_to_direction_proba() first, which converts a predicted
return into a soft direction distribution via a scaled logistic/softmax
rather than a hard sign — so the meta-learner still sees a confidence, not
just a vote.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

OOF_DIR = Path(__file__).resolve().parent.parent / "fusion" / "oof"


def regression_to_direction_proba(pred_return: np.ndarray, scale: float = 5e-4) -> np.ndarray:
    """Maps a predicted (log-)return into a soft 3-class [down, flat, up]
    distribution: treats the predicted return as the logit-scale signal for
    "up vs down", scaled by `scale` (a typical 10s-bar log-return magnitude —
    see label_return_h1's realistic range), then splits out a "flat" mass
    inversely proportional to |pred_return| so a near-zero prediction reads
    as genuinely uncertain rather than forced into up or down."""
    z = pred_return / max(scale, 1e-12)
    p_up_vs_down = 1.0 / (1.0 + np.exp(-z))
    flat_mass = np.clip(1.0 - np.tanh(np.abs(z)), 0.05, 0.9)
    p_up = (1 - flat_mass) * p_up_vs_down
    p_down = (1 - flat_mass) * (1 - p_up_vs_down)
    p_flat = flat_mass
    return np.stack([p_down, p_flat, p_up], axis=1)


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
    name = "regime_states" if member_name == "hmm-regime" else member_name
    path = OOF_DIR / f"{name}.parquet"
    return [path] if path.exists() else []
