from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import joblib
import numpy as np
import pandas as pd

from neuxus.node import Node

# Reuse the attached MI baseline implementation.
from mi_bci.model import FBCSPLDAModel


class Classify(Node):
    """Load a model from a joblib save and classify all input vectors.

    Attributes:
      - output (Port): signal output port

    Args:
      - model_file (str): path to the model file
      - output (str): 'class' or 'probability', specify the output

    Example:
      Classify(port, 'LDA.sav', 'class')
    """

    def __init__(self, input_port, model_file, output):
        Node.__init__(self, input_port)
        self._loaded_model = joblib.load(model_file)

        assert output in ['class', 'probability']
        self._output = output

        if self._output == 'class':
            self._columns = ['class']
        else:
            self._columns = self.input.channels

        self.output.set_parameters(
            data_type='signal',
            channels=self._columns,
            sampling_frequency=self.input.sampling_frequency,
            meta=self.input.meta,
        )

        Node.log_instance(self, {
            'path to model': model_file,
            'model': self._loaded_model,
        })

    def update(self):
        for vector in self.input:
            if self._output == 'class':
                self.output.set(
                    self._loaded_model.predict(vector.values.tolist()),
                    columns=self._columns,
                    timestamps=vector.index,
                )
            else:
                self.output.set(
                    self._loaded_model.predict_proba(vector.values.tolist()),
                    columns=self._columns,
                    timestamps=vector.index,
                )


@dataclass
class _PendingMarker:
    label: int
    start: float
    end: float


class MIModelTrainer(Node):
    """Collect marker-locked MI epochs from a continuous EEG stream and train
    an FBCSP + shrinkage LDA model on termination.

    Input:
      - self.input: continuous EEG signal port
      - self.marker_input: marker port

    Output:
      - none (training side-effect only; the trained model is saved to disk)
    """

    def __init__(
        self,
        input_port,
        marker_input_port,
        model_out: str,
        class_markers: Dict[object, int],
        t_train: Sequence[float] = (0.5, 3.5),
        apply_car_ref: bool = False,
        notch_hz: Optional[float] = None,
        filter_bank: Sequence[Sequence[float]] = ((8, 12), (12, 16), (16, 20), (20, 26), (26, 30)),
        csp_pairs_per_band: int = 2,
        reject_z_thresh: Optional[float] = 5.0,
        min_trials: int = 20,
    ):
        Node.__init__(self, input_port, add_output=False)
        self.marker_input = marker_input_port
        self.model_out = model_out
        self.class_markers = class_markers
        self.t_train = tuple(float(x) for x in t_train)
        self.apply_car_ref = apply_car_ref
        self.notch_hz = notch_hz
        self.filter_bank = tuple(tuple(float(v) for v in band) for band in filter_bank)
        self.csp_pairs_per_band = int(csp_pairs_per_band)
        self.reject_z_thresh = reject_z_thresh
        self.min_trials = int(min_trials)

        self._signal = pd.DataFrame(columns=self.input.channels)
        self._pending = deque()
        self._epochs = []
        self._labels = []

        Node.log_instance(self, {
            'marker port': self.marker_input.id,
            'model_out': self.model_out,
            'class_markers': self.class_markers,
            't_train': self.t_train,
            'filter_bank': self.filter_bank,
            'csp_pairs_per_band': self.csp_pairs_per_band,
        })

    def _append_markers(self):
        for marker_df in self.marker_input:
            for ts in marker_df.index:
                raw = marker_df.loc[ts].values[0]
                key = raw.item() if hasattr(raw, 'item') else raw
                if key in self.class_markers:
                    self._pending.append(
                        _PendingMarker(
                            label=int(self.class_markers[key]),
                            start=float(ts) + self.t_train[0],
                            end=float(ts) + self.t_train[1],
                        )
                    )

    def _extract_ready_epochs(self):
        if self._signal.empty:
            return

        last_t = float(self._signal.index[-1])
        while self._pending and self._pending[0].end <= last_t:
            ev = self._pending.popleft()
            epoch = self._signal[(self._signal.index >= ev.start) & (self._signal.index < ev.end)]
            if epoch.empty:
                continue
            self._epochs.append(epoch.copy())
            self._labels.append(ev.label)

        if self._pending:
            keep_from = self._pending[0].start
            self._signal = self._signal[self._signal.index >= keep_from]
        elif not self._signal.empty:
            self._signal = self._signal.iloc[0:0]

    def update(self):
        self._append_markers()
        for chunk in self.input:
            self._signal = pd.concat([self._signal, chunk])
        self._extract_ready_epochs()

    def terminate(self):
        if len(self._epochs) < self.min_trials:
            print(f"MIModelTrainer: only {len(self._epochs)} epochs collected; not saving a model.")
            return

        fs = float(self.input.sampling_frequency)
        target_len = int(round((self.t_train[1] - self.t_train[0]) * fs))
        X_trials = []
        y = []
        for epoch, label in zip(self._epochs, self._labels):
            arr = epoch[self.input.channels].to_numpy(dtype=float).T
            if arr.shape[1] < target_len:
                continue
            X_trials.append(arr[:, :target_len])
            y.append(label)

        if len(X_trials) < self.min_trials:
            print(f"MIModelTrainer: only {len(X_trials)} valid fixed-length epochs; not saving a model.")
            return

        X = np.stack(X_trials, axis=2)
        y = np.asarray(y, dtype=int)

        model = FBCSPLDAModel(
            fs=fs,
            t_train=(0.0, self.t_train[1] - self.t_train[0]),
            apply_car_ref=self.apply_car_ref,
            notch_hz=self.notch_hz,
            filter_bank=self.filter_bank,
            csp_pairs_per_band=self.csp_pairs_per_band,
            reject_z_thresh=self.reject_z_thresh,
        )
        model.fit(X, y)

        bundle = {
            'model_dict': model.to_dict(),
            'channels': list(self.input.channels),
            'classes': model.classes_.tolist() if model.classes_ is not None else None,
            'training_epochs': int(X.shape[2]),
        }
        joblib.dump(bundle, self.model_out)
        print(f"MIModelTrainer: saved model bundle to {self.model_out}")


class FBCSPClassifier(Node):
    """Apply a trained FBCSP + shrinkage LDA model to incoming epochs.

    Input:
      - epoch port, one NeuXus epoch per iteration

    Output:
      - signal port with one row per epoch and probability columns p_<class>
    """

    def __init__(self, input_port, model_file: str):
        Node.__init__(self, input_port)
        assert self.input.data_type == 'epoch'

        bundle = joblib.load(model_file)
        self.model = FBCSPLDAModel.from_dict(bundle['model_dict'])
        self.classes = list(self.model.lda.classes_)
        self._columns = [f'p_{c}' for c in self.classes]

        self.output.set_parameters(
            data_type='signal',
            channels=self._columns,
            sampling_frequency=self.input.epoching_frequency or self.input.sampling_frequency,
            meta={'source_model': model_file},
        )

        Node.log_instance(self, {
            'model_file': model_file,
            'classes': self.classes,
            'output columns': self._columns,
        })

    def update(self):
        for epoch in self.input:
            X = epoch[self.input.channels].to_numpy(dtype=float).T[:, :, None]
            p = self.model.predict_proba(X)[0].tolist()
            ts = [float(epoch.index[-1])]
            self.output.set([p], ts, self._columns)


class ProbabilitySmoother(Node):
    """Smooth class probabilities and emit a final decision with optional no-decision."""

    def __init__(self, input_port, alpha: float = 0.8, conf_thresh: float = 0.65):
        Node.__init__(self, input_port)
        assert self.input.data_type in ['signal', 'vector']
        self.alpha = float(alpha)
        self.conf_thresh = float(conf_thresh)
        self._prob_cols = list(self.input.channels)
        self._state = np.ones(len(self._prob_cols), dtype=float) / max(len(self._prob_cols), 1)
        self._columns = self._prob_cols + ['decision', 'confidence']

        self.output.set_parameters(
            data_type='signal',
            channels=self._columns,
            sampling_frequency=self.input.sampling_frequency,
            meta=self.input.meta,
        )

        Node.log_instance(self, {
            'alpha': self.alpha,
            'conf_thresh': self.conf_thresh,
        })

    def update(self):
        for chunk in self.input:
            for ts, row in chunk.iterrows():
                p = row[self._prob_cols].to_numpy(dtype=float)
                self._state = self.alpha * self._state + (1.0 - self.alpha) * p
                idx = int(np.argmax(self._state))
                conf = float(self._state[idx])
                decision = 0 if conf < self.conf_thresh else idx + 1
                out = list(self._state) + [decision, conf]
                self.output.set([out], [ts], self._columns)
