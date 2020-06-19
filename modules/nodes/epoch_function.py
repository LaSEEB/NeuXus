import sys

import pandas as pd
import numpy as np

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

        self.value = np.array([0] * len(self.input.channels))

        # TO DO terminate

    def update(self):
        for epoch in self.input:
            mean = epoch.mean()
            self.output.set_from_df(pd.DataFrame(
                mean, columns=[epoch.index[-1]]).transpose())
            self.value = np.array(mean.values)
