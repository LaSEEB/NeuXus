import sys

sys.path.append('..')
import numpy as np

from modules.nodes import (filter, io, select, epoching, epoch_function, store, generate, feature, function, classify)

lsl_reception = generate.Generator('simulation', 16, 500)

chans = select.ChannelSelector(lsl_reception.output, 'index', [1, 2, 3, 4])

butter_filter = filter.ButterFilter(chans.output, 8, 12)

time_epoch = epoching.TimeBasedEpoching(butter_filter.output, 0.5, 0.5)
square_epoch = function.ApplyFunction(time_epoch.output, lambda x: x**2)
average_epoch = epoch_function.Average(square_epoch.output)

log_epoch = function.ApplyFunction(average_epoch.output, lambda x: np.log1p(np.log1p(x)))
features = feature.FeatureAggregator(log_epoch.output)
class_ = classify.Classify(features.output, '../examples/myfile_lda_model.sav')
