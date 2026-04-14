"""
Online NeuXus pipeline for the trained motor-imagery model.

Flow:
LSL EEG -> channel selection -> notch -> CAR -> 2 s epochs every 250 ms
-> FBCSP + LDA probabilities -> smoothing / confidence gate -> LSL send
"""

from neuxus.nodes import io, filter, select, epoching
from neuxus.nodes.classify import FBCSPClassifier, ProbabilitySmoother

EEG_STREAM_TYPE = 'EEG'
MODEL_FILE = 'trained_mi_fbcsp_lda.joblib'
SELECTED_CHANNELS = ['FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C3', 'C1', 'Cz', 'C2', 'C4', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4']

# 1) EEG input
source = io.LslReceive(prop='type', value=EEG_STREAM_TYPE, data_type='signal', sync='network')

# 2) Spatial subset over sensorimotor cortex
chan = select.ChannelSelector(input_port=source.output, mode='name', selected=SELECTED_CHANNELS)

# 3) Minimal preprocessing
notch = filter.NotchFilter(input_port=chan.output, frequency_to_remove=50, quality_factor=35)
car = select.CommonAverageReference(input_port=notch.output)

# 4) Sliding windows: 2 s window, 250 ms hop
epochs = epoching.TimeBasedEpoching(input_port=car.output, duration=2.0, interval=0.25)

# 5) FBCSP + shrinkage LDA probabilities
proba = FBCSPClassifier(input_port=epochs.output, model_file=MODEL_FILE)

# 6) Probability smoothing and no-decision output
decision = ProbabilitySmoother(input_port=proba.output, alpha=0.8, conf_thresh=0.65)

# 7) Publish decoded output over LSL
send = io.LslSend(input_port=decision.output, name='MI_Decoder_Output', type='signal')
