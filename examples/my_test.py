import sys

sys.path.append('..')

from modules.nodes import *

lsl_reception = io.LslReceive('type', 'EEG')  # or (port0, 'type', 'signal')
matrix = {
        'OC2': [1, 1, 0, 1],
        'OC3': [1, 0, 1, 1]}
channel_selector = select.ChannelSelector(lsl_reception.output, 'index', [1, 2, 3, 4])
spatial_filter = select.SpatialFilter(channel_selector.output, matrix='../examples/example')
lsl_send = io.LslSend(spatial_filter.output, 'spatial_filter')
