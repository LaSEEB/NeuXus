import sys
sys.path.append('..')
import numpy as np
from modules.nodes import (filter, io, select, epoching, epoch_function,
                           store, generate, feature, function, classify, display)
import datetime
date = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
print(date)
# data aqcuisition from LSL stream
#lsl_signal = io.LslReceive('name', 'LiveAmpSN-054207-0168', sync='network')
lsl_signal = generate.Generator('simulation', 32, 250)
lsl_markers = io.LslReceive('type', 'Markers', data_type='marker')
# channel selection
chans = select.ChannelSelector(lsl_signal.output, 'index', [8, 25])
# spatial filtering
'''
matrix = {
        'OC2': [1, 1, 0, 1],
        'OC3': [1, 0, 1, 1]}
laplacian_filter =  select.SpatialFilter(chans.output, matrix)
'''
# temporal filtering
butter_filter = filter.ButterFilter(chans.output, 8, 12)
# left epochs
left_epochs = epoching.StimulationBasedEpoching(
    butter_filter.output, lsl_markers.output, 769, 0, 4)
time_epoch_l = epoching.TimeBasedEpoching(
    left_epochs.output, 1, 0.2)  # time-based epoching
square_epoch_l = function.ApplyFunction(
    time_epoch_l.output, lambda x: x**2)  # signal squaring
average_epoch_l = epoch_function.UnivariateStat(
    square_epoch_l.output, 'mean')  # averaging
log_power_l = function.ApplyFunction(
    average_epoch_l.output, lambda x: np.log1p(x))  # logarithmized
# save features
left_features = feature.FeatureAggregator(log_power_l.output, '1')
features_file = '../examples/Epochs_'  # +date
tocsvL = store.ToCsv(left_features.output, features_file)
# right epochs
right_epochs = epoching.StimulationBasedEpoching(
    butter_filter.output, lsl_markers.output, 770, 0, 4)
time_epoch_r = epoching.TimeBasedEpoching(
    right_epochs.output, 1, 0.2)  # time-based epoching
square_epoch_r = function.ApplyFunction(
    time_epoch_r.output, lambda x: x**2)  # signal squaring
average_epoch_r = epoch_function.UnivariateStat(
    square_epoch_r.output, 'mean')  # averaging
log_power_r = function.ApplyFunction(
    average_epoch_r.output, lambda x: np.log1p(x))  # logarithmized
# save features
right_features = feature.FeatureAggregator(log_power_r.output, '2')
tocsvR = store.ToCsv(right_features.output, features_file)
disp1 = display.Plot(log_power_r.output)
#lsl_send = io.LslSend(log_power_l.output, 'log_power_l', 'EEG')
