"""
Calibration / training pipeline for 2-class motor imagery in NeuXus.

This single script can train either:
- from live LSL streams, or
- by replaying a saved dataset through NeuXus `read.Reader`

Supported replay formats follow NeuXus Reader support:
- .xdf
- .gdf
- .set
- .vhdr

How to use
----------
1) Live LSL training
   - set SOURCE_MODE = 'lsl'
   - provide EEG and marker stream types

2) Replay a saved recording in real time
   - set SOURCE_MODE = 'file'
   - set REPLAY_FILE to your .xdf/.gdf/.set/.vhdr file

Stop NeuXus to trigger `terminate()` and save the model bundle.
"""

from neuxus.nodes import io, read, filter, select
from neuxus.nodes.classify import MIModelTrainer

# -----------------------------------------------------------------------------
# Source selection
# -----------------------------------------------------------------------------
# 'lsl'  -> live EEG + markers from LSL
# 'file' -> replay a saved dataset with NeuXus Reader
SOURCE_MODE = 'lsl'

# -----------------------------------------------------------------------------
# Live LSL configuration
# -----------------------------------------------------------------------------
EEG_STREAM_TYPE = 'EEG'
MARKER_STREAM_TYPE = 'Markers'

# -----------------------------------------------------------------------------
# Replay configuration
# -----------------------------------------------------------------------------
# Supported by NeuXus Reader: .xdf, .gdf, .set, .vhdr
REPLAY_FILE = 'path/to/recording.xdf'
REPLAY_MIN_CHUNK_SIZE = 8

# -----------------------------------------------------------------------------
# Training / preprocessing configuration
# -----------------------------------------------------------------------------
MODEL_OUT = 'trained_mi_fbcsp_lda.joblib'
SELECTED_CHANNELS = [
    'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
    'C3', 'C1', 'Cz', 'C2', 'C4',
    'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
]

# Marker mapping for the trainer.
# Adjust this to match your dataset annotations if needed.
CLASS_MARKERS = {
    1: 1,
    2: 2,
    '1': 1,
    '2': 2,
    'LEFT': 1,
    'RIGHT': 2,
    'left': 1,
    'right': 2,
}

NOTCH_HZ = 50
NOTCH_Q = 35
T_TRAIN = (0.0, 2.0)
FILTER_BANK = ((8, 12), (12, 16), (16, 20), (20, 26), (26, 30))
CSP_PAIRS_PER_BAND = 2
REJECT_Z_THRESH = 5.0
MIN_TRIALS = 20

# -----------------------------------------------------------------------------
# Build the source side of the pipeline.
# -----------------------------------------------------------------------------
if SOURCE_MODE == 'lsl':
    eeg_source = io.LslReceive(
        prop='type',
        value=EEG_STREAM_TYPE,
        data_type='signal',
        sync='network',
    )
    marker_source = io.LslReceive(
        prop='type',
        value=MARKER_STREAM_TYPE,
        data_type='marker',
        sync='local',
    )
    eeg_port = eeg_source.output
    marker_port = marker_source.output

elif SOURCE_MODE == 'file':
    replay = read.Reader(REPLAY_FILE, min_chunk_size=REPLAY_MIN_CHUNK_SIZE)
    eeg_port = replay.output
    marker_port = replay.marker_output

else:
    raise ValueError("SOURCE_MODE must be either 'lsl' or 'file'")

# -----------------------------------------------------------------------------
# Common preprocessing and training graph.
# -----------------------------------------------------------------------------
# Sensorimotor channel subset.
chan = select.ChannelSelector(input_port=eeg_port, mode='name', selected=SELECTED_CHANNELS)

# Real-time-safe preprocessing before storing training epochs.
notch = filter.NotchFilter(
    input_port=chan.output,
    frequency_to_remove=NOTCH_HZ,
    quality_factor=NOTCH_Q,
)
car = select.CommonAverageReference(input_port=notch.output)

# Collect cue-locked epochs and fit the model on exit.
trainer = MIModelTrainer(
    input_port=car.output,
    marker_input_port=marker_port,
    model_out=MODEL_OUT,
    class_markers=CLASS_MARKERS,
    t_train=T_TRAIN,
    apply_car_ref=False,   # already done in the NeuXus graph
    notch_hz=None,         # already done in the NeuXus graph
    filter_bank=FILTER_BANK,
    csp_pairs_per_band=CSP_PAIRS_PER_BAND,
    reject_z_thresh=REJECT_Z_THRESH,
    min_trials=MIN_TRIALS,
)
