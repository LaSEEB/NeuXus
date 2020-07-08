import sys

sys.path.append('..')
import numpy as np

from modules.nodes import (filter, io, select, epoching, epoch_function, store, generate, feature, function, classify, display)


# data aqcuisition from LSL stream
lsl_signal = io.LslReceive('name', 'LiveAmpSN-054207-0168', data_type='signal', sync='network')
#lsl_markers = io.LslReceive('type', 'Markers', data_type='marker')

#channel selection
chans = select.ChannelSelector(lsl_signal.output, 'index', [8, 25])

#spatial filtering
'''
matrix = {
        'OC2': [1, 1, 0, 1],
        'OC3': [1, 0, 1, 1]}
laplacian_filter =  select.SpatialFilter(chans.output, matrix)
'''

#temporal filtering
butter_filter = filter.ButterFilter(chans.output, 8, 12)


time_epoch = epoching.TimeBasedEpoching(butter_filter.output, 1, 0.25) #time-based epoching
square_epoch = function.ApplyFunction(time_epoch.output, lambda x: x**2) #signal squaring
average_epoch = epoch_function.UnivariateStat(square_epoch.output, 'mean') #averaging
log_power = function.ApplyFunction(average_epoch.output, lambda x: np.log1p(x)) #logarithmized


features = feature.FeatureAggregator(log_power.output)
mi_class = classify.Classify(features.output, '../examples/motor-imagery-simple/lda_model.sav')
disp1 = display.Plot(mi_class.output)
