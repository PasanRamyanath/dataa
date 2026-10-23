"""
feature engineering scripts/fusion/conformal_dirichlet_calibrator.py
True Diagonal-Preserved Symmetric ODIR Dirichlet Calibrator (Kull et al., NeurIPS 2019)
combined with Online Adaptive Conformal Inference (ACI, Gibbs & Candes 2021) for
non-stationary financial time series.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


class SymmetricODIRDirichletCalibrator:
    """
    Fits a 3-class Dirichlet calibration mapping on the probability simplex:
        p_cal = Softmax(W * log(p_raw) + b)
    Enforces:
      1. True ODIR Regularization: penalizes ONLY off-diagonal cross-terms W_{ij} (i != j)
         and intercepts b_i, leaving diagonal elements W_{ii} unpenalized around 1.0
         so predictions NEVER collapse to the constant base rate!
      2. Up/Down Structural Symmetry: W_{0,0} = W_{2,2}, W_{0,2} = W_{2,0}, b_0 = b_2
         so long-run secular drift cannot induce a permanent LONG or SHORT bias.
      3. Online Adaptive Conformal Inference (ACI): updates effective alpha_t online
         via alpha_{t+1} = clip(alpha_t + gamma * (alpha_target - err_t), 0.10, 0.35).
    """

    def __init__(
        self,
        lambda_odir: float = 1e-2,
        mu_intercept: float = 5e-2,
        aci_gamma: float = 0.015,
    ) -> None:
        self.lambda_odir = lambda_odir
        self.mu_intercept = mu_intercept
        self.aci_gamma = aci_gamma
        self.W_: np.ndarray = np.eye(3, dtype=np.float64)
        self.b_: np.ndarray = np.zeros(3, dtype=np.float64)
        self.q_hat_: float = 0.45
        self.alpha_target_: float = 0.22
        self.alpha_online_: float = 0.22
        self._calib_scores_: np.ndarray = np.array([0.45], dtype=np.float64)

    @staticmethod
    def _unpack_symmetric_params(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w_diag_dir, w_diag_flat, w_cross_dir, w_dir_from_flat, w_flat_from_dir, b_dir, b_flat = theta
        W = np.array([
            [w_diag_dir,      w_dir_from_flat, w_cross_dir],
            [w_flat_from_dir, w_diag_flat,     w_flat_from_dir],
            [w_cross_dir,     w_dir_from_flat, w_diag_dir],
        ], dtype=np.float64)
        b = np.array([b_dir, b_flat, b_dir], dtype=np.float64)
        return W, b

    def fit(
        self,
        raw_probs: np.ndarray,
        y_true: np.ndarray,
        alpha: float = 0.22,
    ) -> "SymmetricODIRDirichletCalibrator":
        self.alpha_target_ = float(alpha)
        self.alpha_online_ = float(alpha)
        eps = 1e-7
        P = np.clip(np.asarray(raw_probs, dtype=np.float64), eps, 1.0 - eps)
        P /= P.sum(axis=1, keepdims=True)
        log_P = np.log(P)
        y = np.asarray(y_true, dtype=np.int64)
        n = len(y)

        split = max(10, int(n * 0.65))
        log_P_fit, y_fit = log_P[:split], y[:split]
        log_P_con, y_con = log_P[split:], y[split:]

        theta0 = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        def _objective(theta: np.ndarray) -> float:
            W, b = self._unpack_symmetric_params(theta)
            logits = log_P_fit @ W.T + b[None, :]
            logits -= logits.max(axis=1, keepdims=True)
            exp_l = np.exp(logits)
            probs = exp_l / exp_l.sum(axis=1, keepdims=True)

            nll = -np.mean(np.log(np.clip(probs[np.arange(len(y_fit)), y_fit], eps, 1.0)))
            off_diag_sq = 2.0 * (theta[2] ** 2 + theta[3] ** 2 + theta[4] ** 2)
            diag_reg = 1e-3 * ((theta[0] - 1.0) ** 2 + (theta[1] - 1.0) ** 2)
            b_sq = 2.0 * (theta[5] ** 2) + (theta[6] ** 2)
            return float(nll + self.lambda_odir * off_diag_sq + diag_reg + self.mu_intercept * b_sq)

        res = minimize(_objective, theta0, method="L-BFGS-B")
        self.W_, self.b_ = self._unpack_symmetric_params(res.x)

        cal_con = self.predict_proba( np.exp(log_P_con) )
        scores = 1.0 - cal_con[np.arange(len(y_con)), y_con]
        self._calib_scores_ = np.sort(scores)
        n_c = len(scores)
        q_level = min(1.0, np.ceil((n_c + 1) * (1.0 - alpha)) / n_c)
        self.q_hat_ = float(np.quantile(scores, q_level))
        return self

    def predict_proba(self, raw_probs: np.ndarray) -> np.ndarray:
        eps = 1e-7
        P = np.clip(np.asarray(raw_probs, dtype=np.float64), eps, 1.0 - eps)
        P /= P.sum(axis=1, keepdims=True)
        logits = np.log(P) @ self.W_.T + self.b_[None, :]
        logits -= logits.max(axis=1, keepdims=True)
        exp_l = np.exp(logits)
        return exp_l / exp_l.sum(axis=1, keepdims=True)

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        return self.predict_proba(raw_probs)

    def update_aci_online(self, realized_class: int, predicted_proba: np.ndarray) -> float:
        """
        Gibbs & Candes (2021) Adaptive Conformal Inference step:
        Adjusts effective miscoverage level alpha_online_ based on whether the realized
        outcome fell inside the current conformal prediction set.
        """
        score = 1.0 - float(predicted_proba[int(realized_class)])
        err = 1.0 if score > self.q_hat_ else 0.0
        self.alpha_online_ = float(
            np.clip(self.alpha_online_ + self.aci_gamma * (self.alpha_target_ - err), 0.10, 0.35)
        )
        if len(self._calib_scores_) > 5:
            q_lvl = float(np.clip(1.0 - self.alpha_online_, 0.50, 0.95))
            self.q_hat_ = float(np.quantile(self._calib_scores_, q_lvl))
        return self.q_hat_

    def is_directional_singleton(self, raw_probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cal_p = self.predict_proba(raw_probs)
        include = (1.0 - cal_p) <= self.q_hat_
        set_size = include.sum(axis=1)
        only_long = (set_size == 1) & include[:, 2]
        only_short = (set_size == 1) & include[:, 0]

        # Dominance fallback if 3-class set is non-singleton due to low SNR regime
        dom_long = (cal_p[:, 2] - cal_p[:, 0] >= 0.18) & (cal_p[:, 2] > cal_p[:, 1])
        dom_short = (cal_p[:, 0] - cal_p[:, 2] >= 0.18) & (cal_p[:, 0] > cal_p[:, 1])

        is_single = only_long | only_short | dom_long | dom_short
        direction = np.where(only_long | dom_long, 1, np.where(only_short | dom_short, -1, 0)).astype(np.int8)
        return is_single, direction
