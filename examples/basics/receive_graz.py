import sys

sys.path.append('../..')

from modules.nodes import (filter, io, select, epoching,
                           epoch_function, store, generate, feature, function, display)

lsl_input_marker = io.LslReceive(
    prop='type',
    value='marker',
    data_type='marker')
graz = display.Graz(
    input_port=lsl_input_marker.output)
