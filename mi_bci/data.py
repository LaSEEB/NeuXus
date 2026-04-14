"""
Utilities for loading MATLAB EEG datasets.

The expected MATLAB format is:
    data.X  -> [channels, samples, trials]
    data.y  -> [trials]
    data.fs -> scalar sampling rate
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
from scipy.io import loadmat


def _unwrap_mat_struct(obj: Any) -> Dict[str, Any]:
    # Handles all common MATLAB struct formats from scipy
    if isinstance(obj, np.ndarray) and obj.size == 1 and obj.dtype == object:
        return _unwrap_mat_struct(obj.flat[0])

    if isinstance(obj, np.ndarray) and obj.dtype.names is not None and obj.size == 1:
        out = {}
        item = obj[0, 0]
        for name in obj.dtype.names:
            out[name] = item[name]
        return out

    if hasattr(obj, "_fieldnames"):
        return {name: getattr(obj, name) for name in obj._fieldnames}

    if isinstance(obj, dict):
        return obj

    raise ValueError(f"Could not parse MATLAB struct named 'data'. Got type: {type(obj)!r}")


def load_mat_data(path: str | Path) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]]:
    path = Path(path)
    mat = loadmat(path, squeeze_me=False, struct_as_record=False)

    if "data" not in mat:
        raise ValueError("MAT-file must contain a struct named 'data'.")

    data = _unwrap_mat_struct(mat["data"])

    if "X" not in data or "y" not in data or "fs" not in data:
        raise ValueError("data struct must contain X, y, and fs.")

    X = np.asarray(data["X"], dtype=float)
    y = np.asarray(data["y"]).reshape(-1).astype(int)
    fs = float(np.asarray(data["fs"]).reshape(-1)[0])

    if X.ndim != 3:
        raise ValueError("data.X must have shape [channels, samples, trials].")
    if X.shape[2] != y.shape[0]:
        raise ValueError("Mismatch between trials and labels.")

    meta = {k: v for k, v in data.items() if k not in {"X", "y", "fs"}}

    return X, y, fs, meta
