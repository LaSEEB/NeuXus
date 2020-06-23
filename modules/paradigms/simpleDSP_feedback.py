import sys

sys.path.append('.')
sys.path.append('../..')

from modules.core.pipeline import run
from modules.nodes import *


if __name__ == '__main__':

    # initialize the pipeline
    lsl_reception = io.LslReceive('name', 'openvibeSignal', sync='from0')  # or (port0, 'type', 'signal')
    select = select.ChannelSelector(lsl_reception.output, 'index', [24])
    butter_filter = filter.ButterFilter(select.output, 8, 12)
    apply_function = function.ApplyFunction(butter_filter.output, lambda x: x**2)
    lsl_send2 = io.LslSend(apply_function.output, 'mySignalFiltered')
    epoch = epoching.TimeBasedEpoching(apply_function.output, 1, 0.5)
    average = epoch_function.Average(epoch.output)
    lsl_send = io.LslSend(average.output, 'mySignalEpoched')

    run()
