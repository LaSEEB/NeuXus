"""
Preprocessing helpers for motor imagery EEG.

This module contains the simple signal processing used by the baseline:
- time-window cropping
- optional power-line notch filtering
- band-pass filtering
- common average reference (CAR)
- simple bad-trial rejection based on robust RMS statistics
"""

from __future__ import annotations

from typing import Optional, Sequence
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def time_to_index(n_samples: int, fs: float, t_win: Sequence[float]) -> tuple[int, int]:
    """
    Convert a time window in seconds into Python sample indices.

    Parameters
    ----------
    n_samples : int
        Number of samples available in the epoch/window.
    fs : float
        Sampling rate in Hz.
    t_win : sequence of float
        Start and end times in seconds, for example (0.5, 3.5).

    Returns
    -------
    (i1, i2) : tuple[int, int]
        Start and end indices suitable for Python slicing `X[:, i1:i2, :]`.

    Notes
    -----
    The end index is exclusive, following normal Python slicing rules.
    """
    # Convert seconds to samples.
    t1 = max(0, int(round(t_win[0] * fs)))
    t2 = min(n_samples, int(round(t_win[1] * fs)))

    if t1 >= t2:
        raise ValueError("Invalid time window.")

    return t1, t2


def apply_notch(X: np.ndarray, fs: float, notch_hz: Optional[float]) -> np.ndarray:
    """
    Apply a notch filter trial-by-trial to suppress power-line interference.

    Parameters
    ----------
    X : np.ndarray
        EEG data with shape [channels, samples, trials].
    fs : float
        Sampling rate in Hz.
    notch_hz : float or None
        Line frequency to suppress, usually 50 or 60 Hz.
        If None, no notch filter is applied.

    Returns
    -------
    np.ndarray
        Filtered EEG array with the same shape as input.
    """
    # Skip filtering entirely when disabled.
    if notch_hz is None:
        return X

    # Design a narrow notch around the target frequency.
    wo = notch_hz / (fs / 2.0)
    bw = wo / 35.0
    b, a = iirnotch(wo, bw)

    # Apply zero-phase filtering trial-by-trial along the time axis.
    Y = np.empty_like(X, dtype=float)
    for tr in range(X.shape[2]):
        Y[:, :, tr] = filtfilt(b, a, X[:, :, tr], axis=1)

    return Y


def design_bandpass(fs: float, f1: float, f2: float, order: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """
    Design a Butterworth band-pass filter.

    Parameters
    ----------
    fs : float
        Sampling rate in Hz.
    f1, f2 : float
        Lower and upper cutoff frequencies in Hz.
    order : int
        Filter order.

    Returns
    -------
    (b, a) : tuple[np.ndarray, np.ndarray]
        IIR filter coefficients for use with `filtfilt`.
    """
    if not (0 < f1 < f2 < fs / 2.0):
        raise ValueError("Invalid bandpass range.")

    return butter(order, [f1 / (fs / 2.0), f2 / (fs / 2.0)], btype="bandpass")


def apply_iir_filter(X: np.ndarray, ba: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """
    Apply an IIR filter trial-by-trial along the time axis.

    Parameters
    ----------
    X : np.ndarray
        EEG data with shape [channels, samples, trials].
    ba : tuple[np.ndarray, np.ndarray]
        Filter coefficients `(b, a)`.

    Returns
    -------
    np.ndarray
        Filtered EEG array.
    """
    b, a = ba
    Y = np.empty_like(X, dtype=float)

    # Process trials independently to keep the implementation simple and explicit.
    for tr in range(X.shape[2]):
        Y[:, :, tr] = filtfilt(b, a, X[:, :, tr], axis=1)

    return Y


def apply_car(X: np.ndarray) -> np.ndarray:
    """
    Apply common average reference (CAR).

    For each time point and trial, subtract the instantaneous mean across channels.
    This is a common and simple spatial preprocessing step for EEG.
    """
    return X - X.mean(axis=0, keepdims=True)


def reject_bad_trials(X: np.ndarray, z_thresh: Optional[float]) -> np.ndarray:
    """
    Reject trials with unusually large overall RMS amplitude.

    Parameters
    ----------
    X : np.ndarray
        EEG data with shape [channels, samples, trials].
    z_thresh : float or None
        Robust z-score threshold. If None, rejection is disabled.

    Returns
    -------
    np.ndarray
        Boolean mask of shape [trials], where True means keep the trial.

    Notes
    -----
    This is deliberately simple. In a production system, you would usually
    combine amplitude checks, channel quality checks, EOG handling, and more.
    """
    if z_thresh is None or not np.isfinite(z_thresh):
        return np.ones(X.shape[2], dtype=bool)

    # Compute one RMS value per trial as a crude artifact score.
    rms = np.sqrt(np.mean(X ** 2, axis=(0, 1)))

    # Use median and MAD for robust scaling.
    med = np.median(rms)
    mad = np.median(np.abs(rms - med)) + np.finfo(float).eps
    rz = 0.6745 * (rms - med) / mad

    good = np.abs(rz) <= z_thresh

    # Safety fallback: do not allow the rejection stage to remove nearly everything.
    if good.sum() < 10:
        return np.ones(X.shape[2], dtype=bool)

    return good
