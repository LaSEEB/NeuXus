from neuxus.nodes import *

generated_data = generate.Generator(
    generator='simulation',
    nb_channels=16,
    sampling_frequency=500)
sending = io.LslSend(
    input_port=generated_data.output,
    name='my_simulated_signal',
    type='signal')
