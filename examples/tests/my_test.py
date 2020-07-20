import sys

import numpy as np

sys.path.append('../..')

from modules.nodes import (read, filter, io, select, epoching, epoch_function, store, generate, feature, function, display, stimulator, processing)

# rda_reception = io.RdaReceive(rdaport=51244, host="192.168.1.132")#, offset=0.125)

'''
xdf = '../../dataset/finger-tapping-graz-protocol.xdf'
gdf = '../../dataset/ME-FG.gdf'
set_ = '../../dataset/ME-FG.gdf_proc.set'
vhdr = '../../dataset/ec-test0005.vhdr'
r = read.Reader(vhdr)
plot = display.Plot(r.output, 5, [1])
gr = display.Graz(r.marker_output)
'''

'''
g = generate.Generator('simulation', 32, 250, min_chunk_size=1)
ds = filter.DownSample(g.output, 5)
plt = display.Plot(ds.output, 5, [1])
plt2 = display.Plot(g.output, 5, [1])
'''

g = generate.Generator('simulation', 4, 250, min_chunk_size=1)
f = filter.ButterFilter(g.output, 1, 120)
t = epoching.TimeBasedEpoching(f.output, 1, 2)
p = processing.Fft(t.output)
plo = display.PlotSpectrum(p.output)
p1 = processing.PsdWelch(t.output)
plo1 = display.PlotSpectrum(p1.output)


#lsl_marker_reception = stimulator.Stimulator('../examples/basic/stim_config.xml')
#d = io.LslSend(lsl_marker_reception.output, 'my stimulus', 'Markers')
#graz_vis = display.Graz(lsl_marker_reception.output)
#my_func = select.SpatialFilter(lsl_reception.output, '../examples/csp_ft.cfg')
'''
lsl_sig_send = io.LslSend(rda_reception.output, 'eeg_data', 'EEG')
lsl_markers_send = io.LslSend(lsl_marker_reception.output, 'ov_markers', 'Markers')'''
'''udp = io.UdpSend(lsl_marker_reception.output, "localhost", 20001)'''

'''chans = select.ChannelSelector(lsl_reception.output, 'index', [1, 2, 3, 4])

butter_filter = filter.ButterFilter(chans.output, 8, 12)

time_epoch = epoching.TimeBasedEpoching(butter_filter.output, 0.5, 0.5)
square_epoch = function.ApplyFunction(time_epoch.output, lambda x: x**2)
average_epoch = epoch_function.Average(square_epoch.output)

log_epoch = function.ApplyFunction(average_epoch.output, lambda x: np.log1p(x))
average_epoch1 = epoching.StimulationBasedEpoching(log_epoch.output, lsl_marker_reception.output, 769, 0, 2)
logpower1 = function.ApplyFunction(average_epoch1.output, lambda x: np.log1p(x))
left_features = feature.FeatureAggregator(logpower1.output, '1')
tocsv = store.ToCsv(left_features.output, 'myfile')
average_epoch2 = epoching.StimulationBasedEpoching(log_epoch.output, lsl_marker_reception.output, 770, 0, 2)
logpower = function.ApplyFunction(average_epoch2.output, lambda x: np.log1p(x))
plt = display.Plot(logpower.output, 10, [1])
right_features = feature.FeatureAggregator(logpower.output, '0')
tocsv2 = store.ToCsv(right_features.output, 'myfile')'''
# featurevector = feature.featureMerger(left_features, right_features, ..., n_features)
'''toCSV(featurevector.output)
loaded_model = joblib.load('lda_model.sav')
loaded_model2 = joblib.load('ann_model.sav')
[[CH1, CH2,... CH_n]]
classify(logpower.output,loaded_model)
classify(logpower.output,loaded_model2)'''
