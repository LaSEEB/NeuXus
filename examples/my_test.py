import sys

sys.path.append('..')

from modules.nodes import (filter, io, select, epoching, epoch_function)

lsl_reception = io.LslReceive('type', 'EEG')  # or (port0, 'type', 'signal')
channel_selector = select.ChannelSelector(lsl_reception.output, 'index', [1, 2, 3, 4])
epoch = epoching.TimeBasedEpoching(channel_selector.output, 1, 0.25)
stat = epoch_function.UnivariateStat(epoch.output, 'iqr', q1=0, q2=1)
stat2 = epoch_function.UnivariateStat(epoch.output, 'range')
lsl_send = io.LslSend(stat.output, 'stat')
lsl_send2 = io.LslSend(stat2.output, 'stat2')
