from neuxus.nodes import *

# dir_path is the path to directory containing this file
stim_cfg = dir_path + '/stimulation_config_1.xml'

generated_markers = stimulator.Stimulator(
    file=stim_cfg)
sending = io.LslSend(
    input_port=generated_markers.output,
    name='my_simulation',
    type='marker')
