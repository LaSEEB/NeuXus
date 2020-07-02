import sys

sys.path.append('..')

from modules.nodes import (filter, io, select, epoching, epoch_function, store, generate)

lsl_reception = io.RdaReceive()
csv = store.ToCsv(lsl_reception.output, 'myCsv')
