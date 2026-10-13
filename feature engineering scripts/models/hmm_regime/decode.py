"""
Causal regime decoding shared by training (OOF regime states) and live
inference (inference/checkpoint_loaders.HmmRegimePredictor).

hmmlearn's model.predict() is Viterbi over the whole sequence it's given and
predict_proba() is forward-backward smoothing — both let bar t+k change the
state assigned to bar t, so decoding a whole val window with them leaks the
future into the stacking meta-learner's regime one-hot. filtered_states()
runs only the forward recursion: state[t] = argmax P(s_t | x_0..x_t).
"""
from __future__ import annotations

import numpy as np


def _log_emission(model, X: np.ndarray) -> np.ndarray:
    """(n, n_states) diagonal-Gaussian log-density, from the fitted
    means_/covars_ only (no private hmmlearn API)."""
    covars = np.asarray(model.covars_)
    var = np.stack([np.diag(c) for c in covars]) if covars.ndim == 3 else covars
    diff = X[:, None, :] - model.means_[None, :, :]
    return -0.5 * (np.log(2 * np.pi * var).sum(axis=1)[None, :] + (diff ** 2 / var[None, :, :]).sum(axis=2))


def filtered_states(model, X_norm: np.ndarray) -> np.ndarray:
    """Per-row argmax of the forward-filtered posterior — each row's state
    depends only on rows at or before it."""
    n = len(X_norm)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    log_b = _log_emission(model, np.asarray(X_norm, dtype=np.float64))
    log_trans = np.log(np.maximum(model.transmat_, 1e-300))
    log_alpha = np.log(np.maximum(model.startprob_, 1e-300)) + log_b[0]
    out = np.empty(n, dtype=np.int64)
    out[0] = int(np.argmax(log_alpha))
    for t in range(1, n):
        prev = log_alpha[:, None] + log_trans
        m = prev.max(axis=0)
        log_alpha = m + np.log(np.exp(prev - m).sum(axis=0)) + log_b[t]
        log_alpha -= log_alpha.max()  # normalize; argmax is unaffected
        out[t] = int(np.argmax(log_alpha))
    return out
