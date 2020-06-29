import sys

sys.path.append('..')

from modules.nodes import (filter, io, select, epoching, epoch_function, store, generate)

lsl_reception = generate.Generator('simulation', 4, 250)  # or (port0, 'type', 'signal')
filter_ = filter.ButterFilter(lsl_reception.output, 8, 12)
tbe = epoching.TimeBasedEpoching(filter_.output, 1, 1)
win = epoch_function.Windowing(tbe.output, 'blackman')
channel_selector = io.LslSend(win.output, 'my_test', type='EEG')
