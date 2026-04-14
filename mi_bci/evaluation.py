"""
Cross-validation and evaluation helpers.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold
from .model import FBCSPLDAModel


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute balanced accuracy for a 2-class or multi-class problem.

    Balanced accuracy is the mean recall across classes and is often more
    informative than plain accuracy when classes are imbalanced.
    """
    classes = np.unique(y_true)
    recalls = []

    for c in classes:
        mask = y_true == c
        recalls.append(np.mean(y_pred[mask] == c) if np.any(mask) else 0.0)

    return float(np.mean(recalls))


def cross_validate_model(X: np.ndarray, y: np.ndarray, model_kwargs: dict, kfold: int = 5, seed: int = 1) -> dict:
    """
    Run stratified k-fold cross-validation for the FBCSP-LDA pipeline.

    Parameters
    ----------
    X : np.ndarray
        EEG array [channels, samples, trials].
    y : np.ndarray
        Labels [trials].
    model_kwargs : dict
        Keyword arguments passed to `FBCSPLDAModel`.
    kfold : int
        Number of folds.
    seed : int
        Random seed for reproducible splitting.

    Returns
    -------
    dict
        Per-fold metrics and their means/standard deviations.
    """
    skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=seed)
    accs = []
    baccs = []

    # `split` ignores X content because the data are already in memory;
    # we only need the label-stratified indices.
    for fold, (tr, te) in enumerate(skf.split(np.zeros_like(y), y), start=1):
        model = FBCSPLDAModel(**model_kwargs)
        model.fit(X[:, :, tr], y[tr])
        yhat = model.predict(X[:, :, te])

        acc = float(np.mean(yhat == y[te]))
        bacc = balanced_accuracy(y[te], yhat)

        accs.append(acc)
        baccs.append(bacc)

        print(f"Fold {fold}: ACC = {acc:.3f}, BACC = {bacc:.3f}")

    return {
        "acc": np.array(accs),
        "bacc": np.array(baccs),
        "mean_acc": float(np.mean(accs)),
        "std_acc": float(np.std(accs)),
        "mean_bacc": float(np.mean(baccs)),
        "std_bacc": float(np.std(baccs)),
    }
