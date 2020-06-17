import sys

sys.path.append('.')
sys.path.append('../..')

from modules.core.pipeline import run
import modules.core.node as node


if __name__ == '__main__':

    # initialize the pipeline
    lsl_reception = node.LslReceive('name', 'openvibeSignal')  # or (port0, 'type', 'signal')
    select = node.ChannelSelector(lsl_reception.output, 'index', [24])
    butter_filter = node.ButterFilter(select.output, 8, 12)
    apply_function = node.ApplyFunction(butter_filter.output, lambda x: x**2)
    lsl_send2 = node.LslSend(apply_function.output, 'mySignalFiltered')
    epoch = node.TimeBasedEpoching(apply_function.output, 1)
    average = node.Averaging(epoch.output)
    lsl_send = node.LslSend(average.output, 'mySignalEpoched')

    run()
