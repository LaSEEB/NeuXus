"""
Online inference helper.

This module keeps a rolling buffer of incoming EEG samples, applies the
trained model to the newest window every fixed step, smooths probabilities,
and produces a final decision or a no-decision output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .model import FBCSPLDAModel


@dataclass
class OnlineBCI:
    """
    Stateful online decoder.

    Parameters
    ----------
    model : FBCSPLDAModel
        Trained offline model.
    win_sec : float
        Classification window length in seconds.
    step_sec : float
        Time between successive classification updates in seconds.
    alpha : float
        Exponential smoothing factor for probabilities.
    conf_thresh : float
        Confidence threshold below which the output becomes 0 (no decision).
    """
    model: FBCSPLDAModel
    win_sec: float = 2.0
    step_sec: float = 0.25
    alpha: float = 0.8
    conf_thresh: float = 0.65

    def __post_init__(self) -> None:
        # Precompute sample counts from timing parameters.
        self.n_win = int(round(self.win_sec * self.model.fs))
        self.n_step = int(round(self.step_sec * self.model.fs))

        # These fields hold the rolling state of the online decoder.
        self.buffer = None
        self.write_pos = 0
        self.total_samples = 0
        self.last_emit = 0

        # Start from an uninformative probability state.
        self.p_smooth = np.array([0.5, 0.5], dtype=float)

    def reset(self) -> None:
        """
        Reset the online state.

        This is useful at trial boundaries in cue-based paradigms.
        """
        self.write_pos = 0
        self.total_samples = 0
        self.last_emit = 0
        self.p_smooth[:] = 0.5

        if self.buffer is not None:
            self.buffer[:] = 0.0

    def _ensure_buffer(self, n_channels: int) -> None:
        """
        Allocate the ring buffer on first use.
        """
        n_buf = max(4 * self.n_win, self.n_win + self.n_step + 10)

        if self.buffer is None:
            self.buffer = np.zeros((n_channels, n_buf), dtype=float)

    def append(self, Xnew: np.ndarray) -> Optional[dict]:
        """
        Append new samples and possibly emit a classification result.

        Parameters
        ----------
        Xnew : np.ndarray
            Incoming EEG chunk with shape [channels, samples].

        Returns
        -------
        dict or None
            None if not enough new data are available yet; otherwise a
            dictionary containing smoothed probabilities, final decision,
            and confidence.
        """
        if Xnew.ndim != 2:
            raise ValueError("Xnew must have shape [channels, samples].")

        self._ensure_buffer(Xnew.shape[0])

        # Push incoming samples into the circular buffer one sample at a time.
        for k in range(Xnew.shape[1]):
            self.write_pos = (self.write_pos + 1) % self.buffer.shape[1]
            self.buffer[:, self.write_pos] = Xnew[:, k]

        self.total_samples += Xnew.shape[1]

        # Do not decode until at least one full window exists.
        if self.total_samples < self.n_win:
            return None

        # Only emit a new decision every `n_step` samples.
        if self.total_samples - self.last_emit < self.n_step:
            return None

        self.last_emit = self.total_samples

        # Read the most recent `n_win` samples from the ring buffer.
        idx = (np.arange(self.write_pos - self.n_win + 1, self.write_pos + 1) % self.buffer.shape[1])
        Xwin = self.buffer[:, idx]

        # The offline model expects [channels, samples, trials], so add a trial axis.
        Xin = Xwin[:, :, None]

        # Predict and smooth the probabilities.
        p = self.model.predict_proba(Xin)[0]
        self.p_smooth = self.alpha * self.p_smooth + (1.0 - self.alpha) * p

        decision_idx = int(np.argmax(self.p_smooth))
        conf = float(self.p_smooth[decision_idx])

        # Output 0 when confidence is too low.
        if conf < self.conf_thresh:
            decision = 0
        else:
            decision = int(self.model.lda.classes_[decision_idx])

        return {
            "prob": self.p_smooth.copy(),
            "decision": decision,
            "confidence": conf,
        }
