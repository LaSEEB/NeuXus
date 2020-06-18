import sys

import pandas as pd

sys.path.append('.')
sys.path.append('../..')

from modules.core.node import Node


class Average(Node):
    """TO DO
    Attributes:
        input_: get DataFrame and meta from input_ port
        output_: output GroupOfPorts
    Args:
        duration: duration of epochs
    """

    def __init__(self, input_port):
        Node.__init__(self, input_port)

        self.output.set_parameters(
            channels=self.input.channels,
            frequency=self.input.frequency,
            meta=self.input.meta)

        # TO DO terminate

    def update(self):
        for epoch in self.input:
            self.output.set_from_df(pd.DataFrame(
                epoch.mean(), columns=[epoch.index[-1]]).transpose())
