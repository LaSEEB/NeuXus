import sys

sys.path.append('..')
import numpy as np

from modules.nodes import (filter, io, select, epoching, epoch_function, store, generate, feature, function, classify, display)

#rda_reception = io.RdaReceive(rdaport=51244, host="192.168.1.132")
rda_reception = io.RdaReceive(rdaport=51244)
plt = display.Plot(rda_reception.output, 1, channels=[1])


#lsl_sendx = io.LslSend(rda_reception.output, 'rda_reception', 'EEG')