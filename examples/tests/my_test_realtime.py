import numpy as np

from neuxus.nodes import *

lsl_reception = generate.Generator('simulation', 16, 500)

chans = select.ChannelSelector(lsl_reception.output, 'index', [1, 2, 3, 4])

butter_filter = filter.ButterFilter(chans.output, 8, 12)

time_epoch = epoching.TimeBasedEpoching(butter_filter.output, 0.5, 0.5)
square_epoch = function.ApplyFunction(time_epoch.output, lambda x: x**2)
average_epoch = epoch_function.UnivariateStat(square_epoch.output, 'mean')

log_epoch = function.ApplyFunction(average_epoch.output, lambda x: np.log1p(np.log1p(x)))
features = feature.FeatureAggregator(log_epoch.output)
class_ = classify.Classify(features.output, '../examples/myfile_lda_model.sav')
# disp1 = display.Plot(class_.output)
