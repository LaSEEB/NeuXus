from neuxus.nodes import *

from printer import Print

generated_data = generate.Generator(
    generator='simulation',
    nb_channels=16,
    sampling_frequency=500
)
pri = Print(
    input_port=generated_data.output
)
