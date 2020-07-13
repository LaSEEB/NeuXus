import sys

sys.path.append('../..')

from modules.nodes import (filter, io, select, epoching,
                           epoch_function, store, generate, feature, function, stimulator)

stim_cfg = '../examples/basic/stim_config.xml'

generated_markers = stimulator.Stimulator(
    file=stim_cfg)
sending = io.LslSend(
    input_port=generated_markers.output,
    name='my_simulation',
    type='marker')
