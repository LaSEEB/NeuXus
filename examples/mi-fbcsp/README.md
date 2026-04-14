# NeuXus conversion for the attached MI-BCI baseline

This folder contains a practical NeuXus conversion of the attached motor-imagery baseline:

- `mi_bci_neuxus_nodes.py`
  - `MIModelTrainer`: collects cue-locked epochs and trains FBCSP + shrinkage LDA on exit
  - `FBCSPClassifier`: loads the saved model and outputs class probabilities from NeuXus epochs
  - `ProbabilitySmoother`: exponential smoothing + confidence threshold + no-decision output
- `pipeline_train_lsl.py`
  - training from either:
    - live LSL EEG + LSL markers, or
    - replay of a saved `.xdf`, `.gdf`, `.set`, or `.vhdr` dataset through `read.Reader`
- `pipeline_online_lsl.py`
  - live online decoding from LSL EEG

## Assumptions
- The attached `mi_bci` package is importable next to the pipeline.
- EEG sampling rate is 250 Hz.
- Marker labels are `1/2` or `LEFT/RIGHT` by default, but the mapping is editable in `CLASS_MARKERS`.
- 15 sensorimotor channels are selected, but you can expand this to 16-32 channels.

## Why custom nodes were needed
NeuXus already provides the right building blocks for:
- LSL IO
- file replay through `Reader`
- notch filtering
- CAR
- channel selection
- time-based epoching

But the attached code depends on learned CSP projections and online probability smoothing. Those are not present as core NeuXus nodes, so they are added here as custom nodes.

## Replay a saved dataset for training
Edit the top of `pipeline_train_lsl.py`:

- set `SOURCE_MODE = 'file'`
- set `REPLAY_FILE` to your recording path

Supported formats are the ones handled by NeuXus `Reader`:
- `.xdf`
- `.gdf`
- `.set`
- `.vhdr`

NeuXus will replay the recording in real time, emit markers from the file, and the trainer will save the model when the pipeline stops.
