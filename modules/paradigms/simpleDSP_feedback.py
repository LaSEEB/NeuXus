import sys

sys.path.append('.')
sys.path.append('../..')

from modules.core.pipeline import run
from modules.nodes import *


if __name__ == '__main__':

    # initialize the pipeline
    lsl_reception = lsl.LslReceive('name', 'openvibeSignal')  # or (port0, 'type', 'signal')
    select = select.ChannelSelector(lsl_reception.output, 'index', [24])
    butter_filter = filter.ButterFilter(select.output, 8, 12)
    apply_function = function.ApplyFunction(butter_filter.output, lambda x: x**2)
    lsl_send2 = lsl.LslSend(apply_function.output, 'mySignalFiltered')
    epoch = epoching.TimeBasedEpoching(apply_function.output, 1)
    average = epoch_function.Average(epoch.output)
    lsl_send = lsl.LslSend(average.output, 'mySignalEpoched')

    run()
