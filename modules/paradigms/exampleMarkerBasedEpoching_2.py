import sys

sys.path.append('.')
sys.path.append('../..')

from modules.core.pipeline import run
from modules.nodes import *


if __name__ == '__main__':

    # get LSL streams
    lsl_marker_reception = lsl.LslReceive('type', 'Markers')
    lsl_reception = lsl.LslReceive('name', 'openvibeSignal')

    # select C3(no.8), C4(no.24) channels
    select = select.ChannelSelector(lsl_reception.output, 'index', [8, 24])

    # spatial filtering
    # ToDo: surface laplacian

    #bandpass filtering
    butter_filter = filter.ButterFilter(select.output, 8, 12)

    # Stimulation-based epoching. 4 second epochs for left:769 and right:770
    stim_epoch_right = epoching.StimulationBasedEpoching(butter_filter.output, lsl_marker_reception.output, 770, 0.5, 4)
    stim_epoch_left = epoching.StimulationBasedEpoching(butter_filter.output, lsl_marker_reception.output, 769, 0.5, 4)

    # time-based epoching every 1 sec.
    time_epoch_right = epoching.TimeBasedEpoching(stim_epoch_right.output, 1)
    time_epoch_left = epoching.TimeBasedEpoching(stim_epoch_left.output, 1)

    # square
    square_epoch_right = function.ApplyFunction(time_epoch_right.output, lambda x: x**2)
    square_epoch_left = function.ApplyFunction(time_epoch_left.output, lambda x: x**2)

    # average
    average_epoch_right = epoch_function.Average(square_epoch_right.output)
    average_epoch_left = epoch_function.Average(square_epoch_left.output)

    #logarithmize
    #log_epoch_right = function.ApplyFunction(average_epoch_right.output, lambda x: log(1+x))
    #log_epoch_left = function.ApplyFunction(average_epoch_left.output, lambda x: log(1+x))

    #feature vector
    #ToDo

    # send via lsl
    lsl_send_right = lsl.LslSend(average_epoch_right.output, 'mySignalEpochedRight')
    lsl_send_left = lsl.LslSend(average_epoch_left.output, 'mySignalEpochedLeft')

    # run the pipeline
    run()
    """
    if len(port2._data) > 0:
        plt.plot(port2._data[0].iloc[:, 0:1].values)
        plt.show()
    if len(port3._data) > 0:
        plt.plot(port3._data[0].iloc[:, 0:1].values)
        plt.show()"""
