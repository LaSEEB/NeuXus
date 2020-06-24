import sys

import logging

sys.path.append('..')

from modules.pipeline import run
from modules.nodes import *


if __name__ == '__main__':
    # numeric_level = getattr(logging, loglevel.upper(), None)
    # if not isinstance(numeric_level, int):
    #    raise ValueError('Invalid log level: %s' % loglevel)
    logging.basicConfig(
        filename='../log/test.log',
        filemode='w',
        format='%(levelname)s %(asctime)s %(message)s',
        level='DEBUG'
    )

    lsl_reception = io.LslReceive('type', 'EEG')  # or (port0, 'type', 'signal')
    butter_filter = filter.ButterFilter(lsl_reception.output, 8, 12)
    apply_function = function.ApplyFunction(butter_filter.output, lambda x: x**2)
    lsl_send2 = io.LslSend(apply_function.output, 'mySignalFiltered')
    epoch = epoching.TimeBasedEpoching(apply_function.output, 1, 0.5)
    average = epoch_function.Average(epoch.output)
    lsl_send = io.LslSend(average.output, 'mySignalEpoched')
    run()
