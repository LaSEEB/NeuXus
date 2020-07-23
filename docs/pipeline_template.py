# description of the pipeline
"""
Detailed template for creating basic NeuXus scripts
Refer to the examples for more complex pipelines and possibilities.

author: Simon Legeay, LaSEEB/CentraleSupélec
mail: simon.legeay.sup@gmail.com

"""

import sys

# import usefull library
import numpy as np
import datetime

# import all nodes from neuxus
# refer to the API to see all available nodes
from neuxus.nodes import *

# add the path to your customized node
sys.path.append('path_to_the_new_module_file')
# import your customized nodes
from my_custom_node_file import MyCustomNode

# this script is exucuted at the very beginning, you can initialze some data, parameters
# and create all nodes of your pipeline

# for example
date = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
print(date)
features_file = 'my_path' + date

# stimulation and visualization
lsl_markers = stimulator.Stimulator(
    file='../utils/stimulus/config_ov.xml'
)

# data aqcuisition from LSL stream
lsl_signal = generate.Generator(
    generator='simulation',
    nb_channels=32,
    sampling_frequency=250
)
# or
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
