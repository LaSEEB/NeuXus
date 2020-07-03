import sys

sys.path.append('../..')

from modules.nodes import (filter, io, select, epoching,
                           epoch_function, store, generate, feature, function)

generated_data = generate.Generator(
    generator='simulation',
    nb_channels=16,
    sampling_frequency=500)
sending = io.LslSend(
    input_port=generated_data.output,
    name='my_simulated_signal',
    type='signal')
