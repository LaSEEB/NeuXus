"""
Main FBCSP + shrinkage LDA model.

The model follows a conventional motor imagery pipeline:
1. Crop the desired time window
2. Optional notch filter
3. Optional common average reference
4. For each frequency band:
   - band-pass filter
   - train CSP
   - compute log-variance features
5. Concatenate all band-wise features
6. Train shrinkage LDA
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Optional, Dict, Any
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from .preprocessing import (
    time_to_index,
    apply_notch,
    design_bandpass,
    apply_iir_filter,
    apply_car,
    reject_bad_trials,
)
from .features import CSPModel, csp_train, csp_features


@dataclass
class FBCSPLDAModel:
    """
    Filter-Bank CSP + shrinkage LDA model for 2-class motor imagery EEG.

    Parameters
    ----------
    fs : float
        Sampling rate in Hz.
    t_train : sequence of float
        Time window in seconds used for both training and inference.
    apply_car_ref : bool
        Whether to apply common average reference.
    notch_hz : float or None
        Power-line notch frequency.
    filter_bank : sequence of (low, high)
        Frequency bands used by the filter bank.
    csp_pairs_per_band : int
        Number of CSP pairs per band.
    reject_z_thresh : float or None
        Robust z-score threshold for training-time bad-trial rejection.
    """
    fs: float
    t_train: Sequence[float] = (0.5, 3.5)
    apply_car_ref: bool = True
    notch_hz: Optional[float] = 50.0
    filter_bank: Sequence[Sequence[float]] = ((8, 12), (12, 16), (16, 20), (20, 26), (26, 30))
    csp_pairs_per_band: int = 2
    reject_z_thresh: Optional[float] = 5.0

    # Learned parameters are stored after fitting.
    band_filters: List[tuple[np.ndarray, np.ndarray]] = field(default_factory=list, init=False)
    csp_models: List[CSPModel] = field(default_factory=list, init=False)
    lda: Optional[LinearDiscriminantAnalysis] = field(default=None, init=False)
    good_trials_: Optional[np.ndarray] = field(default=None, init=False)
    classes_: Optional[np.ndarray] = field(default=None, init=False)

    def _crop(self, X: np.ndarray) -> np.ndarray:
        """
        Crop epochs/windows to the configured analysis interval.
        """
        i1, i2 = time_to_index(X.shape[1], self.fs, self.t_train)
        return X[:, i1:i2, :]

    def _preprocess_for_training(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply preprocessing used during training.

        Returns
        -------
        Xp : np.ndarray
            Preprocessed data after optional rejection.
        good : np.ndarray
            Boolean trial mask indicating which trials were kept.
        """
        X = self._crop(X)
        X = apply_notch(X, self.fs, self.notch_hz)

        # Reject only during training, not at inference.
        good = reject_bad_trials(X, self.reject_z_thresh)
        X = X[:, :, good]

        if self.apply_car_ref:
            X = apply_car(X)

        return X, good

    def _preprocess_for_inference(self, X: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing used during inference.

        Trial rejection is omitted because online prediction must return an output
        for every requested window. Higher-level logic can still suppress weak outputs.
        """
        X = self._crop(X)
        X = apply_notch(X, self.fs, self.notch_hz)

        if self.apply_car_ref:
            X = apply_car(X)

        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FBCSPLDAModel":
        """
        Train the full filter-bank CSP + shrinkage LDA pipeline.

        Parameters
        ----------
        X : np.ndarray
            EEG with shape [channels, samples, trials].
        y : np.ndarray
            Labels with shape [trials].

        Returns
        -------
        FBCSPLDAModel
            The fitted model instance.
        """
        if X.ndim != 3:
            raise ValueError("X must have shape [channels, samples, trials].")

        y = np.asarray(y).reshape(-1).astype(int)

        if X.shape[2] != y.shape[0]:
            raise ValueError("Number of trials does not match labels.")
        if np.unique(y).size != 2:
            raise ValueError("Exactly 2 classes are required.")

        # Shared preprocessing.
        Xp, good = self._preprocess_for_training(X)
        yp = y[good]

        self.good_trials_ = good
        self.classes_ = np.unique(yp)
        self.band_filters = []
        self.csp_models = []
        feat_list = []

        # Train one CSP block per band and concatenate all band-wise features.
        for band in self.filter_bank:
            ba = design_bandpass(self.fs, float(band[0]), float(band[1]))
            Xb = apply_iir_filter(Xp, ba)
            csp = csp_train(Xb, yp, self.csp_pairs_per_band)
            Fb = csp_features(Xb, csp)

            self.band_filters.append(ba)
            self.csp_models.append(csp)
            feat_list.append(Fb)

        F = np.concatenate(feat_list, axis=1)

        # `solver="lsqr", shrinkage="auto"` gives a practical shrinkage LDA baseline.
        self.lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        self.lda.fit(F, yp)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform EEG epochs/windows into the final concatenated feature matrix.

        Parameters
        ----------
        X : np.ndarray
            EEG with shape [channels, samples, trials].

        Returns
        -------
        np.ndarray
            Feature matrix with shape [trials, total_features].
        """
        if self.lda is None or not self.band_filters or not self.csp_models:
            raise RuntimeError("Model has not been fitted.")

        Xp = self._preprocess_for_inference(X)
        feat_list = []

        for ba, csp in zip(self.band_filters, self.csp_models):
            Xb = apply_iir_filter(Xp, ba)
            feat_list.append(csp_features(Xb, csp))

        return np.concatenate(feat_list, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for each trial/window.
        """
        if self.lda is None:
            raise RuntimeError("Model has not been fitted.")

        F = self.transform(X)
        return self.lda.predict_proba(F)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for each trial/window.
        """
        if self.lda is None:
            raise RuntimeError("Model has not been fitted.")

        F = self.transform(X)
        return self.lda.predict(F)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Return the LDA decision score for each trial/window.
        """
        if self.lda is None:
            raise RuntimeError("Model has not been fitted.")

        F = self.transform(X)
        score = self.lda.decision_function(F)
        return np.asarray(score).reshape(-1)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the fitted model into a serializable dictionary.

        This is used together with `joblib.dump`.
        """
        return {
            "fs": self.fs,
            "t_train": tuple(self.t_train),
            "apply_car_ref": self.apply_car_ref,
            "notch_hz": self.notch_hz,
            "filter_bank": [tuple(b) for b in self.filter_bank],
            "csp_pairs_per_band": self.csp_pairs_per_band,
            "reject_z_thresh": self.reject_z_thresh,
            "good_trials_": self.good_trials_,
            "classes_": self.classes_,
            "band_filters": self.band_filters,
            "csp_models": self.csp_models,
            "lda": self.lda,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FBCSPLDAModel":
        """
        Restore a fitted model from a dictionary created by `to_dict`.
        """
        obj = cls(
            fs=d["fs"],
            t_train=d["t_train"],
            apply_car_ref=d["apply_car_ref"],
            notch_hz=d["notch_hz"],
            filter_bank=d["filter_bank"],
            csp_pairs_per_band=d["csp_pairs_per_band"],
            reject_z_thresh=d["reject_z_thresh"],
        )
        obj.good_trials_ = d["good_trials_"]
        obj.classes_ = d["classes_"]
        obj.band_filters = d["band_filters"]
        obj.csp_models = d["csp_models"]
        obj.lda = d["lda"]
        return obj
