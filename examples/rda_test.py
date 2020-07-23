import sys

sys.path.append('..')
import numpy as np

from sygnal.nodes import (filter, io, select, epoching, epoch_function, store, generate, feature, function, classify, display)

#rda_reception = io.RdaReceive(rdaport=51244, host="192.168.1.132")
rda_reception = io.RdaReceive(rdaport=51244)
plt = display.Plot(rda_reception.output, 5, channels=[1, 2, 3, 4, 5])
csv = store.ToCsv(rda_reception.marker_output, 'RDAmarkers')


#lsl_sendx = io.LslSend(rda_reception.output, 'rda_reception', 'EEG')