"""
feature engineering scripts/fusion/supervised_multi_horizon_stacker.py
Remediates Audit Flaw #1: Replaces heuristic probability blending with a
genuinely trained Supervised Multi-Horizon OOF Stacker + LightGBM Expected
Net-PnL Meta-Labeler fitted on time-purged Out-of-Fold (OOF) predictions.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from fusion.conformal_dirichlet_calibrator import SymmetricODIRDirichletCalibrator
from fusion.hierarchical_fusion_v3 import (
    compute_cross_horizon_alignment_features,
    shannon_negentropy_norm,
)


class SupervisedMultiHorizonOOFStacker:
    """
    Fits a supervised hierarchical meta-stacker and expected-net-PnL LightGBM
    meta-labeler on the causally aligned multi-tier OOF Parquet store.
    Never uses hardcoded heuristic weights when trained artifacts are present!
    """

    def __init__(self, c_reg: float = 0.25, conformal_alpha: float = 0.22) -> None:
        self.c_reg = c_reg
        self.conformal_alpha = conformal_alpha
        self.direction_stacker = LogisticRegression(
            C=c_reg,
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
        self.dirichlet_calibrator = SymmetricODIRDirichletCalibrator(
            lambda_odir=1e-2, mu_intercept=5e-2
        )
        self.meta_win_clf: lgb.LGBMClassifier | None = None
        self.meta_net_bps_reg: lgb.LGBMRegressor | None = None
        self.meta_isotonic: IsotonicRegression | None = None
        self.is_fitted_: bool = False

    @staticmethod
    def build_meta_feature_matrix(
        tier4_macro_probs: np.ndarray,     # (N, 3) aggregated or concatenated Tier 4 OOF probs
        tier3_meso_probs: np.ndarray,      # (N, 3) aggregated or concatenated Tier 3 OOF probs
        tier2_tactical_probs: np.ndarray,  # (N, 3) aggregated or concatenated Tier 2 OOF probs
        tier1_micro_probs: np.ndarray,     # (N, 3) aggregated or concatenated Tier 1 OOF probs
        all_member_probs_flat: np.ndarray, # (N, 3 * M_members) full member OOF matrix
        hmm_regime_probs: np.ndarray,      # (N, 4) 4-state HMM regime posteriors
        atr_15m_bps: np.ndarray,           # (N,)
        atr_1h_bps: np.ndarray,            # (N,)
    ) -> np.ndarray:
        """
        Constructs the supervised meta-feature tensor X_meta combining:
          - Individual member OOF probabilities across all 4 tiers
          - Pairwise Cross-Horizon Alignment vectors phi_align(3,4), phi_align(2,3), phi_align(1,2), phi_align(2,4)
          - Tier-level Shannon negentropies (certainty scores)
          - 4-state HMM regime posteriors and volatility term-structure ratio (ATR_15m / ATR_1h)
        """
        align_3_4 = compute_cross_horizon_alignment_features(tier3_meso_probs, tier4_macro_probs)
        align_2_3 = compute_cross_horizon_alignment_features(tier2_tactical_probs, tier3_meso_probs)
        align_2_4 = compute_cross_horizon_alignment_features(tier2_tactical_probs, tier4_macro_probs)
        align_1_2 = compute_cross_horizon_alignment_features(tier1_micro_probs, tier2_tactical_probs)

        cert_4 = shannon_negentropy_norm(tier4_macro_probs)[:, None]
        cert_3 = shannon_negentropy_norm(tier3_meso_probs)[:, None]
        cert_2 = shannon_negentropy_norm(tier2_tactical_probs)[:, None]
        cert_1 = shannon_negentropy_norm(tier1_micro_probs)[:, None]

        vol_ratio = (atr_15m_bps / np.maximum(atr_1h_bps * 0.5, 1.0))[:, None]

        return np.hstack([
            all_member_probs_flat,
            tier4_macro_probs,
            tier3_meso_probs,
            tier2_tactical_probs,
            tier1_micro_probs,
            align_3_4,
            align_2_3,
            align_2_4,
            align_1_2,
            cert_4,
            cert_3,
            cert_2,
            cert_1,
            hmm_regime_probs,
            vol_ratio,
        ]).astype(np.float32)

    def fit_on_purged_oof(
        self,
        X_meta: np.ndarray,
        y_tb_3class: np.ndarray,          # {0: SHORT, 1: FLAT, 2: LONG} from fee_floored_triple_barrier_labels_v4
        net_ret_long_bps: np.ndarray,     # Realized net bps if LONG
        net_ret_short_bps: np.ndarray,    # Realized net bps if SHORT
        sample_weights: np.ndarray | None = None,
        purge_bars: int = 16,             # 16 x 15m = 4 hours strict purge between Stacker fit and Calibrator/Meta fit
    ) -> dict[str, float]:
        """
        Fits the 2-stage supervised stacker + ODIR Dirichlet Calibrator + LightGBM
        Expected-Net-PnL Meta-Labeler using a strict chronological split with purge_bars.
        """
        n = len(y_tb_3class)
        if n < 200:
            raise ValueError(f"Insufficient OOF rows ({n}) to fit SupervisedMultiHorizonOOFStacker")

        split_1 = int(n * 0.55)
        split_2 = min(n - 50, split_1 + purge_bars)

        X_tr, y_tr = X_meta[:split_1], y_tb_3class[:split_1]
        w_tr = sample_weights[:split_1] if sample_weights is not None else None

        X_cal, y_cal = X_meta[split_2:], y_tb_3class[split_2:]
        net_l_cal = net_ret_long_bps[split_2:]
        net_s_cal = net_ret_short_bps[split_2:]

        # 1. Fit Primary Direction Stacker on [0 .. split_1]
        self.direction_stacker.fit(X_tr, y_tr, sample_weight=w_tr)

        # 2. Generate OOF raw probabilities on [split_2 .. n] and fit Symmetric ODIR Dirichlet Calibrator
        raw_cal_probs = self.direction_stacker.predict_proba(X_cal)
        self.dirichlet_calibrator.fit(raw_cal_probs, y_cal, alpha=self.conformal_alpha)
        cal_probs = self.dirichlet_calibrator.predict_proba(raw_cal_probs)

        # 3. Construct Meta-Label Targets on [split_2 .. n]:
        # Did the primary stacker's chosen side (LONG vs SHORT) make > +2.0 bps NET of fees?
        chosen_long = cal_probs[:, 2] >= cal_probs[:, 0]
        realized_chosen_net_bps = np.where(chosen_long, net_l_cal, net_s_cal)
        y_meta_win = (realized_chosen_net_bps > 2.0).astype(np.int32)

        # Augment X_cal with calibrated stacker probabilities and conviction margin
        margin = np.abs(cal_probs[:, 2] - cal_probs[:, 0])[:, None]
        X_meta_stage2 = np.hstack([X_cal, cal_probs, margin])

        cal_mid = int(len(y_cal) * 0.70)
        cal_mid_purged = min(len(y_cal) - 20, cal_mid + purge_bars)

        self.meta_win_clf = lgb.LGBMClassifier(
            n_estimators=180,
            max_depth=4,
            num_leaves=12,
            learning_rate=0.03,
            subsample_for_bin=5000,
            min_child_samples=40,
            colsample_bytree=0.8,
            subsample=0.8,
            reg_alpha=1.5,
            reg_lambda=2.5,
            random_state=42,
            verbose=-1,
        )
        self.meta_win_clf.fit(X_meta_stage2[:cal_mid], y_meta_win[:cal_mid])

        self.meta_net_bps_reg = lgb.LGBMRegressor(
            objective="huber",
            alpha=1.35,
            n_estimators=180,
            max_depth=4,
            num_leaves=12,
            learning_rate=0.03,
            min_child_samples=40,
            reg_alpha=1.5,
            reg_lambda=2.5,
            random_state=42,
            verbose=-1,
        )
        self.meta_net_bps_reg.fit(X_meta_stage2[:cal_mid], realized_chosen_net_bps[:cal_mid])

        # Fit Isotonic Calibrator on held-out tail [cal_mid_purged:]
        raw_meta_p = self.meta_win_clf.predict_proba(X_meta_stage2[cal_mid_purged:])[:, 1]
        self.meta_isotonic = IsotonicRegression(y_min=0.02, y_max=0.98, out_of_bounds="clip")
        self.meta_isotonic.fit(raw_meta_p, y_meta_win[cal_mid_purged:])
        self.is_fitted_ = True

        # Evaluate Holdout Precision on Conformal + Meta-Gated Trades
        cal_meta_p = self.meta_isotonic.transform(raw_meta_p)
        pred_net_bps = self.meta_net_bps_reg.predict(X_meta_stage2[cal_mid_purged:])
        act_mask = (cal_meta_p >= 0.58) & (pred_net_bps >= 6.0)
        actual_net_holdout = realized_chosen_net_bps[cal_mid_purged:]

        n_trades = int(np.sum(act_mask))
        win_rate = float(np.mean(actual_net_holdout[act_mask] > 0.0) * 100.0) if n_trades > 0 else 0.0
        mean_net = float(np.mean(actual_net_holdout[act_mask])) if n_trades > 0 else 0.0

        return {
            "oof_holdout_authorized_trades": float(n_trades),
            "oof_holdout_net_win_rate_pct": round(win_rate, 2),
            "oof_holdout_avg_net_bps": round(mean_net, 2),
        }

    def save(self, dest_dir: Path) -> None:
        dest_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, dest_dir / "supervised_multi_horizon_stacker.joblib")

    @classmethod
    def load(cls, dest_dir: Path) -> "SupervisedMultiHorizonOOFStacker":
        return joblib.load(dest_dir / "supervised_multi_horizon_stacker.joblib")
