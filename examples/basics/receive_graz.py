from neuxus.nodes import *

lsl_input_marker = io.LslReceive(
    prop='type',
    value='marker',
    data_type='marker')
graz = display.Graz(
    input_port=lsl_input_marker.output)
