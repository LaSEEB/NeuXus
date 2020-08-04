from neuxus.nodes import *


def square(x):
    # x is a matrix of a row
    return x ** 2


lsl_reception = io.LslReceive(
    prop='type',
    value='EEG',
    data_type='signal'
)

butter_filter = filter.ButterFilter(
    input_port=lsl_reception.output,
    lowcut=8,
    highcut=12
)

apply_function = function.ApplyFunction(
    input_port=butter_filter.output,
    function=square
)

epoch = epoching.TimeBasedEpoching(
    input_port=apply_function.output,
    duration=1,
    interval=0.5
)

average = epoch_function.UnvariateStat(
    input_port=epoch.output,
    stat='mean'
)

lsl_send = io.LslSend(
    input_port=average.output,
    name='mySignalEpoched'
)
