import sys

sys.path.append('..')
import numpy as np

from modules.nodes import (processing, filter, io, select, epoching, epoch_function, store, generate, feature, function, display, stimulator)

# data aqcuisition from LSL stream
lsl_signal = generate.Generator('simulation', 32, 250)

#channel selection
selected_chans = select.ChannelSelector(lsl_signal.output, 'index', [1, 2])

butter_filter = filter.ButterFilter(selected_chans.output, 1, 4)

time_epoch = epoching.TimeBasedEpoching(butter_filter.output, 3, 0.5) #time-based epoching

hilb = processing.HilbertTransform(time_epoch.output)


#disp0 = display.Plot(selected_chans.output)
disp1 = display.Plot(time_epoch.output)
disp2 = display.Plot(hilb.output)
