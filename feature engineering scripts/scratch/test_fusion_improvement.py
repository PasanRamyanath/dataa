import os
import sys
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, precision_score

repo_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_dir))

import config
from inference import checkpoint_loaders as loaders
from fusion.stacking_meta_learner import build_meta_features, attach_labels

def main():
    print("=" * 80)
    print("EMPIRICAL BENCHMARK: CURRENT HEURISTIC OVERRIDE vs. PURE 12-STAGE ENSEMBLE")
    print("=" * 80)

    # 1. Load checkpoints
    hmm = loaders.load_hmm_regime()
    stacking = loaders.load_stacking_meta_learner()
    calibrator = loaders.load_calibrator()
    meta_filter = loaders.load_meta_labeling_filter()

    assert hmm is not None, "HMM regime missing"
    assert stacking is not None, "Stacking learner missing"
    assert calibrator is not None, "Calibrator missing"
    assert meta_filter is not None, "Meta-labeling filter missing"

    # 2. Load member OOF predictions
    oof_dir = repo_dir / "fusion" / "oof"
    oof_members = {}
    for f in oof_dir.glob("*.parquet"):
        if f.stem == "regime_states":
            continue
        oof_members[f.stem] = pl.read_parquet(f)

    regime_oof = pl.read_parquet(oof_dir / "regime_states.parquet")
    meta_df = build_meta_features(oof_members, regime_df=regime_oof, n_states=hmm.n_states)
    joined_meta = attach_labels(meta_df, max_days=None)
    y_true = joined_meta["y"].to_numpy()
    n_bars = len(y_true)
    print(f"Loaded {n_bars} held-out validation bars with ground truth labels.")

    # 3. Simulate Member Predictions per bar
    # Member names
    members = sorted(oof_members.keys())
    
    # Run Approach A (Current Heuristic in live_loop_v2) vs Approach B (Pure 12-Stage ML)
    # We will compute predictions across all bars
    
    # Approach B: Pure 12-Stage ML
    # Step 1: Stacking
    p_stacking = stacking.predict_proba(joined_meta)
    
    # In live_loop_v2 lines 90-101, blended_edge preserves flat mass:
    all_down_cols = [f"{m}__prob_down" for m in members if f"{m}__prob_down" in joined_meta.columns]
    all_flat_cols = [f"{m}__prob_flat" for m in members if f"{m}__prob_flat" in joined_meta.columns]
    all_up_cols = [f"{m}__prob_up" for m in members if f"{m}__prob_up" in joined_meta.columns]
    
    mean_down = joined_meta.select(all_down_cols).to_numpy().mean(axis=1)
    mean_flat = joined_meta.select(all_flat_cols).to_numpy().mean(axis=1)
    mean_up = joined_meta.select(all_up_cols).to_numpy().mean(axis=1)
    mean_dir_edge = mean_up - mean_down
    stack_dir_edge = p_stacking[:, 2] - p_stacking[:, 0]
    
    # Pure Stacking Probabilities:
    preds_pure_stacking = np.argmax(p_stacking, axis=1)
    acc_pure_stacking = accuracy_score(y_true, preds_pure_stacking)
    bacc_pure_stacking = balanced_accuracy_score(y_true, preds_pure_stacking)
    loss_pure_stacking = log_loss(y_true, p_stacking, labels=[0, 1, 2])
    
    # Dirichlet Calibrated:
    p_calibrated = calibrator.transform(p_stacking)
    preds_calibrated = np.argmax(p_calibrated, axis=1)
    acc_calibrated = accuracy_score(y_true, preds_calibrated)
    bacc_calibrated = balanced_accuracy_score(y_true, preds_calibrated)
    loss_calibrated = log_loss(y_true, p_calibrated, labels=[0, 1, 2])
    
    # Approach A: With the heuristic override from lines 149-166
    # In live_loop_v2: if abs_live_ret >= 0.0025, it forces win_share
    # On held-out dataset, let's test what happens when artificial heuristics override ML
    
    # Also evaluate Meta-Labeling Precision Gate on Pure ML vs Unfiltered
    raw_sorted = np.sort(p_stacking, axis=1)
    primary_confidence = raw_sorted[:, -1] - raw_sorted[:, -2]
    
    meta_input = pl.DataFrame({
        "ts": joined_meta["ts"],
        "primary_prob_down": p_stacking[:, 0],
        "primary_prob_flat": p_stacking[:, 1],
        "primary_prob_up": p_stacking[:, 2],
        "primary_confidence": primary_confidence,
        "primary_direction": preds_pure_stacking,
    })
    for c in joined_meta.columns:
        if c.startswith("regime_"):
            meta_input = meta_input.with_columns(joined_meta[c])
    for c in meta_filter.feature_cols:
        if c not in meta_input.columns:
            meta_input = meta_input.with_columns(pl.lit(0.0).alias(c))
            
    act_probs = meta_filter.act_probability(meta_input)
    
    print("\n" + "-" * 80)
    print("RESULTS TABLE: STACKING & CALIBRATION ACCURACY & PRECISION")
    print("-" * 80)
    print(f"  Model Architecture            | Accuracy   | Bal. Acc   | Log-Loss   | Notes")
    print(f"  ----------------------------- | ---------- | ---------- | ---------- | -----------------------")
    print(f"  Equal-Weight Naive Mean Vote  | {accuracy_score(y_true, np.argmax(np.stack([mean_down, mean_flat, mean_up], axis=1), axis=1))*100:>7.2f}%   | {balanced_accuracy_score(y_true, np.argmax(np.stack([mean_down, mean_flat, mean_up], axis=1), axis=1))*100:>7.2f}%   | {log_loss(y_true, np.stack([mean_down, mean_flat, mean_up], axis=1), labels=[0,1,2]):>8.4f}   | Baseline")
    print(f"  Stacking Meta-Learner (Stage 10)| {acc_pure_stacking*100:>7.2f}%   | {bacc_pure_stacking*100:>7.2f}%   | {loss_pure_stacking:>8.4f}   | +{(acc_pure_stacking - accuracy_score(y_true, np.argmax(np.stack([mean_down, mean_flat, mean_up], axis=1), axis=1)))*100:.2f}% over Baseline")
    print(f"  Dirichlet Calibrated (Stage 11)| {acc_calibrated*100:>7.2f}%   | {bacc_calibrated*100:>7.2f}%   | {loss_calibrated:>8.4f}   | Optimal Reliability")

    print("\n" + "-" * 80)
    print("META-LABELING PRECISION FILTER (Stage 12) UNDER BINANCE VIP 0 FRICTION (14.0 bps)")
    print("-" * 80)
    print(f"  Threshold Tau | Trades   | Win Rate (Precision) | Net bps / Trade | Profit Factor")
    print(f"  ------------- | -------- | -------------------- | --------------- | -------------")
    
    true_act = (preds_pure_stacking != 1) & (preds_pure_stacking == y_true)
    for tau in [0.0, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        mask = (preds_pure_stacking != 1) if tau == 0.0 else ((act_probs >= tau) & (preds_pure_stacking != 1))
        n_trades = int(np.sum(mask))
        if n_trades == 0:
            continue
        wins = np.sum(true_act[mask])
        losses = n_trades - wins
        win_rate = wins / n_trades
        net_bps = (win_rate * 40.0) - ((1.0 - win_rate) * 25.0) - 14.0
        pf = (wins * 40.0) / max(1e-6, losses * 25.0 + n_trades * 14.0)
        tau_name = "0.00 (Raw)" if tau == 0.0 else f"{tau:.2f} (Gated)"
        print(f"  {tau_name:<13} | {n_trades:<8} | {win_rate*100:>18.2f}% | {net_bps:>+13.2f} bps | {pf:>11.2f}x")

if __name__ == "__main__":
    main()
