import sys
import numpy as np

sys.path.append('.')
sys.path.append('../..')

from modules.core.pipeline import run
from modules.nodes import *


if __name__ == '__main__':

    # initialize the pipeline
    lsl_marker_reception = lsl.LslReceive('type', 'Markers')
    lsl_reception = lsl.LslReceive('name', 'LiveAmpSN-054207-0168')
    #lsl_reception = lsl.LslReceive('name', 'openvibeSignal')

    select = select.ChannelSelector(lsl_reception.output, 'index', [8, 25])

    butter_filter = filter.ButterFilter(select.output, 8, 12)

    time_epoch = epoching.TimeBasedEpoching(butter_filter.output, 1, 0.5)

    square_epoch = function.ApplyFunction(time_epoch.output, lambda x: x**2)

    average_epoch = epoch_function.Average(square_epoch.output)

    #log_epoch = function.ApplyFunction(time_epoch.output, lambda x: np.log1p(x))

    baseline = epoching.StimulationBasedEpoching(average_epoch.output, lsl_marker_reception.output, 770, 0.125, 3)
    baseline_average = epoch_function.Average(baseline.output)


    def relative_alpha(x):
        return (x - baseline_average.value) / baseline_average.value * 100


    relative_band = function.ApplyFunction(average_epoch.output, relative_alpha)


    lsl_send = lsl.LslSend(relative_band.output, 'relative_band')

    # run the pipeline
    run()
