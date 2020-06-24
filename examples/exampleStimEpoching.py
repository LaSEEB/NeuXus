import sys

sys.path.append('.')
sys.path.append('../..')

from modules.core.pipeline import run
from modules.nodes import *


if __name__ == '__main__':

    # initialize the pipeline
    lsl_marker_reception = io.LslReceive('type', 'Markers')
    lsl_reception = io.LslReceive('name', 'openvibeSignal')
    select = select.ChannelSelector(lsl_reception.output, 'index', [23, 24])
    epoch_right = epoching.StimulationBasedEpoching(select.output, lsl_marker_reception.output, 770, 0.125, 1)
    epoch_right_average = epoch_function.Average(epoch_right.output)

    epoch_left = epoching.StimulationBasedEpoching(select.output, lsl_marker_reception.output, 769, 0.125, 1)

    def my_function(x):
        return x + epoch_right_average.value

    epoch_left_f = function.ApplyFunction(epoch_left.output, my_function)

    lsl_send = lsl.LslSend(epoch_left_f.output, 'mySignalEpoched')

    # run the pipeline
    run()
    """
    if len(port2._data) > 0:
        plt.plot(port2._data[0].iloc[:, 0:1].values)
        plt.show()
    if len(port3._data) > 0:
        plt.plot(port3._data[0].iloc[:, 0:1].values)
        plt.show()"""
