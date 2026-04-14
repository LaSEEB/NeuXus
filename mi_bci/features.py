"""
Feature extraction for motor imagery EEG.

This module implements Common Spatial Patterns (CSP) and the standard
log-variance CSP feature transform.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh


@dataclass
class CSPModel:
    """
    Container for a trained CSP projection matrix.

    Attributes
    ----------
    W : np.ndarray
        CSP spatial filters with shape [channels, selected_components].
    classes : np.ndarray
        The two class labels seen during training.
    """
    W: np.ndarray
    classes: np.ndarray


def csp_train(X: np.ndarray, y: np.ndarray, n_pairs: int = 2) -> CSPModel:
    """
    Train a 2-class Common Spatial Patterns projection.

    Parameters
    ----------
    X : np.ndarray
        Band-limited EEG with shape [channels, samples, trials].
    y : np.ndarray
        Labels with shape [trials].
    n_pairs : int
        Number of CSP pairs to keep. The output dimension becomes 2 * n_pairs.

    Returns
    -------
    CSPModel
        Trained CSP projection.

    Notes
    -----
    CSP finds spatial filters that maximize variance for one class while
    minimizing it for the other. The most discriminative filters appear at
    both ends of the eigen-spectrum, so we keep symmetric pairs.
    """
    classes = np.unique(y)
    if classes.size != 2:
        raise ValueError("CSP requires exactly 2 classes.")

    n_ch = X.shape[0]
    C1 = np.zeros((n_ch, n_ch), dtype=float)
    C2 = np.zeros((n_ch, n_ch), dtype=float)
    n1 = 0
    n2 = 0

    # Estimate one normalized covariance per trial and average by class.
    for i in range(X.shape[2]):
        Xi = X[:, :, i]
        Ci = Xi @ Xi.T

        # Normalize by trace so trial scale does not dominate covariance averaging.
        Ci = Ci / max(np.trace(Ci), np.finfo(float).eps)

        if y[i] == classes[0]:
            C1 += Ci
            n1 += 1
        else:
            C2 += Ci
            n2 += 1

    C1 /= max(n1, 1)
    C2 /= max(n2, 1)

    # Very small Tikhonov regularization improves numerical stability.
    reg = 1e-6 * np.trace(C1 + C2) / n_ch
    C1 += reg * np.eye(n_ch)
    C2 += reg * np.eye(n_ch)

    # Solve the generalized eigenvalue problem:
    #   C1 w = lambda (C1 + C2) w
    eigvals, W = eigh(C1, C1 + C2)

    # Largest and smallest eigenvalues correspond to the most class-selective filters.
    order = np.argsort(eigvals)[::-1]
    W = W[:, order]

    # Keep symmetric pairs from both extremes.
    sel = list(range(n_pairs)) + list(range(W.shape[1] - n_pairs, W.shape[1]))
    Wsel = W[:, sel]

    return CSPModel(W=Wsel, classes=classes)


def csp_features(X: np.ndarray, csp: CSPModel) -> np.ndarray:
    """
    Compute log-variance CSP features for each trial.

    Parameters
    ----------
    X : np.ndarray
        Band-limited EEG with shape [channels, samples, trials].
    csp : CSPModel
        Trained CSP projection.

    Returns
    -------
    np.ndarray
        Feature matrix with shape [trials, n_csp_features].
    """
    n_trials = X.shape[2]
    n_feat = csp.W.shape[1]
    F = np.zeros((n_trials, n_feat), dtype=float)

    for i in range(n_trials):
        # Project the trial into CSP space.
        Z = csp.W.T @ X[:, :, i]

        # Variance per CSP component becomes the base feature.
        v = np.var(Z, axis=1, ddof=0)

        # Normalize total variance and take log, which is standard for CSP.
        v = v / max(v.sum(), np.finfo(float).eps)
        F[i, :] = np.log(v)

    return F
