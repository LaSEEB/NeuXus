from neuxus.nodes import *

lsl_reception = io.LslReceive('type', 'signal')  # or (port0, 'type', 'signal')
butter_filter = filter.ButterFilter(lsl_reception.output, 8, 12)
apply_function = function.ApplyFunction(butter_filter.output, lambda x: x**2)
lsl_send2 = io.LslSend(apply_function.output, 'mySignalFiltered')
epoch = epoching.TimeBasedEpoching(apply_function.output, 1, 0.5)
average = epoch_function.Average(epoch.output)
lsl_send = io.LslSend(average.output, 'mySignalEpoched')
