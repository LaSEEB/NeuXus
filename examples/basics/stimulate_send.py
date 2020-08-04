from neuxus.nodes import *

# dir_path is the path to directory containing this file
stim_cfg = dir_path + '/stimulation_config_1.xml'

# Node 1
generated_markers = stimulator.Stimulator(
    file=stim_cfg
)

# Node 2
sending = io.LslSend(
    input_port=generated_markers.output,
    name='my_simulation',
    type='marker'
)
