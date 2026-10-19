import json
import os
import sys
from pathlib import Path
import urllib.request
import numpy as np
import polars as pl
from datetime import datetime, timezone

repo_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_dir))

from inference import checkpoint_loaders as loaders
from fusion.stacking_meta_learner import build_meta_features

def test_live_feed():
    url = "https://analyses-organized-enables-mechanism.trycloudflare.com/latest"
    print("Fetching live tick from Kaggle VM...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=8) as resp:
        data = json.loads(resp.read().decode())["latest"]

    print(f"Bar Time: {data.get('datetime')} | Close: ${data.get('close')}")
    print(f"Reported in Payload:")
    print(f"  - Proba: Down={data.get('prob_down'):.4f}, Flat={data.get('prob_flat'):.4f}, Up={data.get('prob_up'):.4f}")
    print(f"  - Primary Dir: {data.get('primary_direction')} | Confidence: {data.get('confidence'):.4f}")
    print(f"  - Act Probability: {data.get('act_probability')} | Bet Size: {data.get('bet_size')}")

    # Now let's run the pure ML fusion locally on the member predictions
    hmm = loaders.load_hmm_regime()
    stacking = loaders.load_stacking_meta_learner()
    calibrator = loaders.load_calibrator()
    meta_filter = loaders.load_meta_labeling_filter()

    m_preds = data.get("member_predictions", {})
    regime_state = data.get("regime_state", 1)

    ts0 = [datetime(2000, 1, 1, tzinfo=timezone.utc)]
    oof_dict = {
        name: pl.DataFrame({
            "ts": ts0,
            "prob_down": [float(p[0])],
            "prob_flat": [float(p[1])],
            "prob_up": [float(p[2])],
        })
        for name, p in m_preds.items()
    }
    regime_df = pl.DataFrame({"ts": ts0, "regime_state": [regime_state]}) if regime_state is not None else None
    meta_row = build_meta_features(oof_dict, regime_df, n_states=hmm.n_states)

    # Missing column handling
    missing = [c for c in stacking.feature_cols if c not in meta_row.columns]
    if missing:
        add_exprs = []
        for c in missing:
            if "__prob_" in c:
                add_exprs.append(pl.lit(1.0/3.0).cast(pl.Float64).alias(c))
            elif c.startswith("regime_"):
                add_exprs.append(pl.lit(0).cast(pl.Int8).alias(c))
            else:
                add_exprs.append(pl.lit(0.0).cast(pl.Float64).alias(c))
        meta_row = meta_row.with_columns(add_exprs)

    # Pure Stacking
    stack_p = stacking.predict_proba(meta_row)[0]
    print(f"\nPure Stacking Meta-Learner Output:")
    print(f"  - Proba: Down={stack_p[0]:.4f}, Flat={stack_p[1]:.4f}, Up={stack_p[2]:.4f}")
    print(f"  - Dir: {np.argmax(stack_p)} | Dir Edge (Up - Down): {stack_p[2] - stack_p[0]:+.4f}")

    # Dirichlet Calibrated
    cal_p = calibrator.transform(stack_p[None, :])[0]
    print(f"Dirichlet Calibrated Output:")
    print(f"  - Proba: Down={cal_p[0]:.4f}, Flat={cal_p[1]:.4f}, Up={cal_p[2]:.4f}")
    print(f"  - Dir: {np.argmax(cal_p)} | Dir Edge (Up - Down): {cal_p[2] - cal_p[0]:+.4f}")

    # Meta-Labeling Filter
    raw_sorted = np.sort(stack_p)
    conf = float(raw_sorted[-1] - raw_sorted[-2])
    p_dir = int(np.argmax(stack_p))
    regime_cols = {f"regime_{s}": [1 if s == regime_state else 0] for s in range(hmm.n_states)}
    row = pl.DataFrame({
        "ts": ts0, **regime_cols,
        "primary_prob_down": [float(stack_p[0])],
        "primary_prob_flat": [float(stack_p[1])],
        "primary_prob_up": [float(stack_p[2])],
        "primary_confidence": [conf],
        "primary_direction": [p_dir],
    })
    missing_mf = [c for c in meta_filter.feature_cols if c not in row.columns]
    if missing_mf:
        row = row.with_columns([pl.lit(0.0).alias(c) for c in missing_mf])
    act_p = float(meta_filter.act_probability(row)[0])
    print(f"Meta-Labeling Filter:")
    print(f"  - Raw Act Probability: {act_p:.4f}")

if __name__ == "__main__":
    test_live_feed()
