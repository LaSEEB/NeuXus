# description of the pipeline
"""
Detailed template for creating basic NeuXus scripts
Refer to the examples for more complex pipelines and possibilities.

author: ...
mail: ...

"""

# import usefull library
import numpy as np
import datetime

# import all nodes from neuxus
# refer to the API to see all available nodes
from neuxus.nodes import *

# import your customized nodes from a file located in the same directory as your pipeline script
from my_custom_node_file import MyCustomNode

# this script is executed at the very beginning, you can initialze some data, parameters
# and create all Nodes you need for your pipeline

# for example
date = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
print(date)
features_file = 'my_path' + date

# stimulation and visualization
lsl_markers = stimulator.Stimulator(
    file=dir_path + 'config_ov.xml'  # dir_path is the path to the directory containing the current script
)

# data aqcuisition from LSL stream
lsl_signal = generate.Generator(
    generator='simulation',
    nb_channels=32,
    sampling_frequency=250
)
# or from simulation
lsl_signal = io.LslReceive(
    prop='name',
    value='LiveAmpSN-054207-0168',
    data_type='signal',
    sync='network'
)

# to link node, use the attribute output in input_port arg of another node

# channel selection
selected_chans = select.ChannelSelector(
    input_port=lsl_signal.output,
    mode='index',
    selected=[8, 25]
)

# temporal filtering
butter_filter = filter.ButterFilter(
    input_port=selected_chans.output,
    lowcut=8,
    highcut=12
)

# time-based epoching
time_epoch = epoching.TimeBasedEpoching(
    input_port=butter_filter.output,
    duration=1,
    interval=0.25
)

# averaging
average_epoch = epoch_function.UnivariateStat(
    input_port=time_epoch.output,
    stat='mean'
)

# save features
tocsvL = store.ToCsv(
    input_port=average_epoch.output,
    file=features_file
)

# my custom node
my_node = MyCustomNode(
    input_port=average_epoch.output
)
