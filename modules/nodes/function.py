import sys

sys.path.append('.')
sys.path.append('../..')

from modules.core.node import Node


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
