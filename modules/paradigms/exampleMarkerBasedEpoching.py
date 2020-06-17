import sys

sys.path.append('.')
sys.path.append('../..')

from modules.core.pipeline import run
import modules.core.node as node


if __name__ == '__main__':

    # initialize the pipeline
    lsl_marker_reception = node.LslReceive('type', 'Markers')
    lsl_reception = node.LslReceive('name', 'openvibeSignal')
    select = node.ChannelSelector(lsl_reception.output, 'index', [24])
    epoch_right = node.StimulationBasedEpoching(select.output, lsl_marker_reception.output, 770, 0.125, 1)
    epoch_left = node.StimulationBasedEpoching(select.output, lsl_marker_reception.output, 769, 0.125, 1)
    lsl_send = node.LslSend(epoch_right.output, 'mySignalEpoched')

    # run the pipeline
    run()
    """
    if len(port2._data) > 0:
        plt.plot(port2._data[0].iloc[:, 0:1].values)
        plt.show()
    if len(port3._data) > 0:
        plt.plot(port3._data[0].iloc[:, 0:1].values)
        plt.show()"""
