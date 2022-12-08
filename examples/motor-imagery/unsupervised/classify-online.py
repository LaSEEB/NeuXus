import numpy as np
from neuxus.nodes import read, display, io, filter, select, epoching, function, epoch_function, log
import argparse
import subprocess

sfol = ''  # Folder to save results to
fil = 'sub-02_ses-lab1_task-neurowMINF'  # File name to save results as (and if reading from a file, to load data from)

# If acquiring data by LSL
# MARKERS = {
#     'show_cross': 5,
#     'show_left_arrow': 7,
#     'show_rigth_arrow': 8,
#     'hide_arrow': 9,
#     'hide_cross': 10,
#     'exit_': 12,
# }

# Receive data from Recorder
# signal = io.RdaReceive(rdaport=51244)

# Generate stimulators
# markers = stimulator.Stimulator(file='/stimulations.xml')

# OPTIONALLY, read data from file:
MARKERS = {
    'show_cross': 'Stimulus/S  5',
    'show_rigth_arrow': 'Stimulus/S  8',
    'show_left_arrow': 'Stimulus/S  7',
    'hide_arrow': 'Stimulus/S  9',
    'hide_cross': 'Stimulus/S 10',
    'exit_': 'Stimulus/S 12',
}
# Read data
signal = read.Reader(fil + '.vhdr')

# Display markers in Graz protocol
marker_display = display.Graz(signal.marker_output)    # When reading from a file
# marker_display = display.Graz(markers.output)  # When receiving from a stream

# Temporal + spatial filter
signal_filt = filter.ButterFilter(signal.output, 1, 30)
signal_car = select.CommonAverageReference(signal.output)
signal_chans = select.ChannelSelector(signal_car.output, 'index', [8, 25])
signal_filt = filter.ButterFilter(signal_chans.output, 8, 12)

# Baseline
base = epoching.StimulationBasedEpoching(signal_filt.output, signal.marker_output, MARKERS['show_cross'], 0, 5)  # When reading from a file
# base = epoching.StimulationBasedEpoching(signal_filt.output, markers.output, MARKERS['show_cross'], 0, 5)    # When receiving from a stream
base_squared = function.ApplyFunction(base.output, lambda x: x**2)
base_mean = epoch_function.UnivariateStat(base_squared.output, 'mean')
base_epoch = epoching.TimeBasedEpoching(base_squared.output, 1, 0.25)  # Once base(_squared) is complete, it will be epoched at once

# 'Task'
epoch = epoching.TimeBasedEpoching(signal_filt.output, 1, 0.25)
epoch_squared = function.ApplyFunction(epoch.output, lambda x: x**2)

# Average each epoch for next metrics
base_epoch_mean = epoch_function.UnivariateStat(base_epoch.output, 'mean')
epoch_mean = epoch_function.UnivariateStat(epoch_squared.output, 'mean')

# ERD%
erd_base_res = function.ApplyFunction(base_epoch_mean.output, lambda x, y: (((x/y.value-1)*100)<-10).astype(int), base_mean)  # [1]
erd_task_res = function.ApplyFunction(epoch_mean.output, lambda x, y: (((x/y.value-1)*100)<-10).astype(int), base_mean)

# Stream out
# lsl_base_res = io.LslSend(input_port=base_res.output, name='base_res', type="epoch", format="int32")
lsl_res = io.LslSend(input_port=erd_task_res.output, name='res', type="epoch", format="int32")

# Save into file
log_erd_10_base = log.Mat(erd_base_res.output, sfol + fil + '_pro-erd10base')
log_erd_10_task = log.Mat(erd_task_res.output, sfol + fil + '_pro-erd10task')

# [1] erd_base_res contains the baseline epochs normalized by the baseline itself.
# Since the baseline must pass to be used, the epochs will be created (at once) right after the baseline has passed.
# For online classification (during the baseline period), these epoch will therefore not be availabe in real-time.
# You can either: use the task epochs (during the baseline) but these will be normalized by the previous baseline;
# or don't use baseline epochs at all, and just use the classifier during the task.
