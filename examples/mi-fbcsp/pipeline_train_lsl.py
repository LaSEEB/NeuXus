"""
Live calibration / training pipeline for 2-class motor imagery in NeuXus.

Markers are expected on a second LSL stream, for example:
- 1 -> left hand MI
- 2 -> right hand MI

Stop NeuXus to trigger `terminate()` and save the model bundle.
"""

from neuxus.nodes import io, filter, select
from neuxus.nodes.classify import MIModelTrainer

EEG_STREAM_TYPE = 'EEG'
MARKER_STREAM_TYPE = 'Markers'
MODEL_OUT = 'trained_mi_fbcsp_lda.joblib'
SELECTED_CHANNELS = ['FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C3', 'C1', 'Cz', 'C2', 'C4', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4']

# Receive EEG and markers from LSL.
eeg = io.LslReceive(prop='type', value=EEG_STREAM_TYPE, data_type='signal', sync='network')
markers = io.LslReceive(prop='type', value=MARKER_STREAM_TYPE, data_type='marker', sync='local')

# Sensorimotor channel subset.
chan = select.ChannelSelector(input_port=eeg.output, mode='name', selected=SELECTED_CHANNELS)

# Online-safe preprocessing before storing training epochs.
notch = filter.NotchFilter(input_port=chan.output, frequency_to_remove=50, quality_factor=35)
car = select.CommonAverageReference(input_port=notch.output)

# Collect 2 s epochs after each cue marker and fit the attached model on exit.
trainer = MIModelTrainer(
    input_port=car.output,
    marker_input_port=markers.output,
    model_out=MODEL_OUT,
    class_markers={1: 1, 2: 2, 'LEFT': 1, 'RIGHT': 2},
    t_train=(0.0, 2.0),
    apply_car_ref=False,   # already done in the NeuXus graph
    notch_hz=None,         # already done in the NeuXus graph
    filter_bank=((8, 12), (12, 16), (16, 20), (20, 26), (26, 30)),
    csp_pairs_per_band=2,
    reject_z_thresh=5.0,
    min_trials=20,
)
