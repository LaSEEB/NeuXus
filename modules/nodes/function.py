import sys

import numpy as np

sys.path.append('../..')

from modules.node import Node


class ApplyFunction(Node):
    """TO DO
    Attributes:
        input_: get DataFrame and meta from input_ port
        output_: output GroupOfPorts
    Args:
        duration: duration of epochs
    """

    def __init__(self, input_port, function):
        Node.__init__(self, input_port)

        self.output.set_parameters(
            channels=self.input.channels,
            frequency=self.input.frequency,
            meta=self.input.meta)

        self.function = function

        # TO DO terminate

    def update(self):
        for chunk in self.input:
            self.output.set_from_df(chunk.apply(self.function, axis=1, raw=True))


class ApplyFunctionFromValue(Node):
    """TO DO
    Attributes:
        input_: get DataFrame and meta from input_ port
        output_: output GroupOfPorts
    Args:
        duration: duration of epochs
    """

    def __init__(self, input_port, port_value, function):
        Node.__init__(self, input_port)

        self.output.set_parameters(
            channels=self.input.channels,
            frequency=self.input.frequency,
            meta=self.input.meta)

        self.function = function
        self.port_value = port_value

        # TO DO terminate

    def update(self):
        for chunk in self.input:
            print(self.port_value.value)
            self.output.set_from_df(chunk.apply(self.function, args=(4, 2), axis=1, raw=True))
            print(chunk.apply(self.function, axis=1, raw=True))
