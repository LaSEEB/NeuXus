import sys

sys.path.append('.')
sys.path.append('../..')

from modules.core.pipeline import run
from modules.nodes import *


if __name__ == '__main__':

    # initialize the pipeline
    lsl_marker_reception = lsl.LslReceive('type', 'Markers')
    lsl_reception = lsl.LslReceive('name', 'openvibeSignal')
    select = select.ChannelSelector(lsl_reception.output, 'index', [24])
    epoch_right = epoching.StimulationBasedEpoching(select.output, lsl_marker_reception.output, 770, 0.125, 1)
    epoch_left = epoching.StimulationBasedEpoching(select.output, lsl_marker_reception.output, 769, 0.125, 1)
    lsl_send = lsl.LslSend(epoch_right.output, 'mySignalEpoched')

    # run the pipeline
    run()
    """
    if len(port2._data) > 0:
        plt.plot(port2._data[0].iloc[:, 0:1].values)
        plt.show()
    if len(port3._data) > 0:
        plt.plot(port3._data[0].iloc[:, 0:1].values)
        plt.show()"""
