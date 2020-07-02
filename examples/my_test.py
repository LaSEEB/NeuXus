import sys

sys.path.append('..')

from modules.nodes import (filter, io, select, epoching, epoch_function, store, generate)

lsl_reception = io.RdaReceive()
butter_filter = filter.ButterFilter(lsl_reception.output, 8, 12)
csv = store.ToCsv(lsl_reception.output, 'myCsv')
