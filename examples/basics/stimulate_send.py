from neuxus.nodes import *

stim_cfg = '../examples/basic/stim_config.xml'

generated_markers = stimulator.Stimulator(
    file=stim_cfg)
sending = io.LslSend(
    input_port=generated_markers.output,
    name='my_simulation',
    type='marker')
