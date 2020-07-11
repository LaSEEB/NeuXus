import sys

sys.path.append('..')
import numpy as np

from modules.nodes import (filter, io, select, epoching, epoch_function, store, generate, feature, function, display, stimulator)

import datetime
date = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
print (date)
features_file = '../examples/motor-imagery-simple/Epochs_'+date

#stimulation and visualization
lsl_markers = stimulator.Stimulator('../utils/stimulus/config_ov.xml') # load config
send_lsl_markers = io.LslSend(lsl_markers.output, 'my_stimulus', 'Markers') # send via LSL
graz_vis = display.Graz(lsl_markers.output) # visualize

# data aqcuisition from LSL stream
lsl_signal = generate.Generator('simulation', 32, 250)
#lsl_signal = io.LslReceive('name', 'LiveAmpSN-054207-0168', data_type='signal', sync='network')
#rda_reception = io.RdaReceive(rdaport=51244, host="192.168.1.132")#, offset=0.125)
#lsl_markers = io.LslReceive('type', 'Markers', data_type='marker') # receive markers from the network

#channel selection
selected_chans = select.ChannelSelector(lsl_signal.output, 'index', [8, 25])

#spatial filtering
'''
matrix = {
        'OC2': [1, 1, 0, 1],
        'OC3': [1, 0, 1, 1]}
laplacian_filter =  select.SpatialFilter(selected_chans.output, matrix)
'''
# load CSP weights from config file
#csp_filter = select.SpatialFilter(selected_chans.output, '../examples/csp_ft.cfg')

#temporal filtering
butter_filter = filter.ButterFilter(selected_chans.output, 8, 12)

#left epochs
left_epochs = epoching.StimulationBasedEpoching(butter_filter.output, lsl_markers.output, 769, 0, 4)
time_epoch_l = epoching.TimeBasedEpoching(left_epochs.output, 1, 0.25) #time-based epoching
square_epoch_l = function.ApplyFunction(time_epoch_l.output, lambda x: x**2) #signal squaring
average_epoch_l = epoch_function.UnivariateStat(square_epoch_l.output, 'mean') #averaging
log_power_l = function.ApplyFunction(average_epoch_l.output, lambda x: np.log1p(x)) #logarithmized
#save features
left_features = feature.FeatureAggregator(log_power_l.output, '1')
tocsvL = store.ToCsv(left_features.output, features_file)


#right epochs
right_epochs = epoching.StimulationBasedEpoching(butter_filter.output, lsl_markers.output, 770, 0, 4)
time_epoch_r = epoching.TimeBasedEpoching(right_epochs.output, 1, 0.25) #time-based epoching
square_epoch_r = function.ApplyFunction(time_epoch_r.output, lambda x: x**2) #signal squaring
average_epoch_r = epoch_function.UnivariateStat(square_epoch_r.output, 'mean') #averaging
log_power_r = function.ApplyFunction(average_epoch_r.output, lambda x: np.log1p(x)) #logarithmized
#save features
right_features = feature.FeatureAggregator(log_power_r.output, '2')
tocsvR = store.ToCsv(right_features.output, features_file)


#disp1 = display.Plot(left_features.output)

lsl_send = io.LslSend(lsl_signal.output, 'lsl_signal', 'EEG')
